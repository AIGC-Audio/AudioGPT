from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors
import torch.optim
import torch.utils.data
import torch
from packaging import version

class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):  # pragma: no cover
        if version.parse(torch.__version__[:6]) < version.parse("1.11"):
            self._sync_params()
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            assert len(self.device_ids) == 1
            if self.module.training:
                output = self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                output = self.module.test_step(*inputs[0], **kwargs[0])
            else:
                output = self.module.validation_step(*inputs[0], **kwargs[0])
            if torch.is_grad_enabled():
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters:
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
        else:
            from torch.nn.parallel.distributed import \
                logging, Join, _DDPSink, _tree_flatten_with_rref, _tree_unflatten_with_rref
            with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.logger.set_runtime_stats_and_log()
                    self.num_iterations += 1
                    self.reducer.prepare_for_forward()

                # Notify the join context that this process has not joined, if
                # needed
                work = Join.notify_join_context(self)
                if work:
                    self.reducer._set_forward_pass_work_handle(
                        work, self._divide_by_initial_world_size
                    )

                # Calling _rebuild_buckets before forward compuation,
                # It may allocate new buckets before deallocating old buckets
                # inside _rebuild_buckets. To save peak memory usage,
                # call _rebuild_buckets before the peak memory usage increases
                # during forward computation.
                # This should be called only once during whole training period.
                if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                    logging.info("Reducer buckets have been rebuilt in this iteration.")
                    self._has_rebuilt_buckets = True

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                buffer_hook_registered = hasattr(self, 'buffer_hook')
                if self._check_sync_bufs_pre_fwd():
                    self._sync_buffers()

                if self._join_config.enable:
                    # Notify joined ranks whether they should sync in backwards pass or not.
                    self._check_global_requires_backward_grad_sync(is_joined_rank=False)

                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])

                # sync params according to location (before/after forward) user
                # specified as part of hook, if hook was specified.
                if self._check_sync_bufs_post_fwd():
                    self._sync_buffers()

                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.require_forward_param_sync = True
                    # We'll return the output object verbatim since it is a freeform
                    # object. We need to find any tensors in this object, though,
                    # because we need to figure out which parameters were used during
                    # this forward pass, to ensure we short circuit reduction for any
                    # unused parameters. Only if `find_unused_parameters` is set.
                    if self.find_unused_parameters and not self.static_graph:
                        # Do not need to populate this for static graph.
                        self.reducer.prepare_for_backward(list(_find_tensors(output)))
                    else:
                        self.reducer.prepare_for_backward([])
                else:
                    self.require_forward_param_sync = False

            # TODO: DDPSink is currently enabled for unused parameter detection and
            # static graph training for first iteration.
            if (self.find_unused_parameters and not self.static_graph) or (
                    self.static_graph and self.num_iterations == 1
            ):
                state_dict = {
                    'static_graph': self.static_graph,
                    'num_iterations': self.num_iterations,
                }

                output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(
                    output
                )
                output_placeholders = [None for _ in range(len(output_tensor_list))]
                # Do not touch tensors that have no grad_fn, which can cause issues
                # such as https://github.com/pytorch/pytorch/issues/60733
                for i, output in enumerate(output_tensor_list):
                    if torch.is_tensor(output) and output.grad_fn is None:
                        output_placeholders[i] = output

                # When find_unused_parameters=True, makes tensors which require grad
                # run through the DDPSink backward pass. When not all outputs are
                # used in loss, this makes those corresponding tensors receive
                # undefined gradient which the reducer then handles to ensure
                # param.grad field is not touched and we don't error out.
                passthrough_tensor_list = _DDPSink.apply(
                    self.reducer,
                    state_dict,
                    *output_tensor_list,
                )
                for i in range(len(output_placeholders)):
                    if output_placeholders[i] is None:
                        output_placeholders[i] = passthrough_tensor_list[i]

                # Reconstruct output data structure.
                output = _tree_unflatten_with_rref(
                    output_placeholders, treespec, output_is_rref
                )
        return output
