import os
import traceback
from functools import partial
from tqdm import tqdm


def chunked_worker(worker_id, args_queue=None, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    while True:
        args = args_queue.get()
        if args == '<KILL>':
            return
        job_idx, map_func, arg = args
        try:
            map_func_ = partial(map_func, ctx=ctx) if ctx is not None else map_func
            if isinstance(arg, dict):
                res = map_func_(**arg)
            elif isinstance(arg, (list, tuple)):
                res = map_func_(*arg)
            else:
                res = map_func_(arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))


class MultiprocessManager:
    def __init__(self, num_workers=None, init_ctx_func=None, multithread=False, queue_max=-1):
        if multithread:
            from multiprocessing.dummy import Queue, Process
        else:
            from multiprocessing import Queue, Process
        if num_workers is None:
            num_workers = int(os.getenv('N_PROC', os.cpu_count()))
        self.num_workers = num_workers
        self.results_queue = Queue(maxsize=-1)
        self.jobs_pending = []
        self.args_queue = Queue(maxsize=queue_max)
        self.workers = []
        self.total_jobs = 0
        self.multithread = multithread
        for i in range(num_workers):
            if multithread:
                p = Process(target=chunked_worker,
                            args=(i, self.args_queue, self.results_queue, init_ctx_func))
            else:
                p = Process(target=chunked_worker,
                            args=(i, self.args_queue, self.results_queue, init_ctx_func),
                            daemon=True)
            self.workers.append(p)
            p.start()

    def add_job(self, func, args):
        if not self.args_queue.full():
            self.args_queue.put((self.total_jobs, func, args))
        else:
            self.jobs_pending.append((self.total_jobs, func, args))
        self.total_jobs += 1

    def get_results(self):
        self.n_finished = 0
        while self.n_finished < self.total_jobs:
            while len(self.jobs_pending) > 0 and not self.args_queue.full():
                self.args_queue.put(self.jobs_pending[0])
                self.jobs_pending = self.jobs_pending[1:]
            job_id, res = self.results_queue.get()
            yield job_id, res
            self.n_finished += 1
        for w in range(self.num_workers):
            self.args_queue.put("<KILL>")
        for w in self.workers:
            w.join()

    def close(self):
        if not self.multithread:
            for w in self.workers:
                w.terminate()

    def __len__(self):
        return self.total_jobs


def multiprocess_run_tqdm(map_func, args, num_workers=None, ordered=True, init_ctx_func=None,
                          multithread=False, queue_max=-1, desc=None):
    for i, res in tqdm(
            multiprocess_run(map_func, args, num_workers, ordered, init_ctx_func, multithread,
                             queue_max=queue_max),
            total=len(args), desc=desc):
        yield i, res


def multiprocess_run(map_func, args, num_workers=None, ordered=True, init_ctx_func=None, multithread=False,
                     queue_max=-1):
    """
    Multiprocessing running chunked jobs.

    Examples:
    >>> for res in tqdm(multiprocess_run(job_func, args):
    >>>     print(res)

    :param map_func:
    :param args:
    :param num_workers:
    :param ordered:
    :param init_ctx_func:
    :param q_max_size:
    :param multithread:
    :return:
    """
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
        # num_workers = 1
    manager = MultiprocessManager(num_workers, init_ctx_func, multithread, queue_max=queue_max)
    for arg in args:
        manager.add_job(map_func, arg)
    if ordered:
        n_jobs = len(args)
        results = ['<WAIT>' for _ in range(n_jobs)]
        i_now = 0
        for job_i, res in manager.get_results():
            results[job_i] = res
            while i_now < n_jobs and (not isinstance(results[i_now], str) or results[i_now] != '<WAIT>'):
                yield i_now, results[i_now]
                results[i_now] = None
                i_now += 1
    else:
        for job_i, res in manager.get_results():
            yield job_i, res
    manager.close()
