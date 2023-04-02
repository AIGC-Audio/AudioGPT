import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GatedGraphConv

def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def group_hidden_by_segs(h, seg_ids, max_len):
    """
    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    # assert h_gby_segs.shape[-1] == 192
    return h_gby_segs

class GraphAuxEnc(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_iterations=5, n_edge_types=6):
        super(GraphAuxEnc, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.skip_connect = True
        self.dropout_after_gae = False

        self.ggc_1 = GatedGraphConv(in_feats=in_dim, out_feats=hid_dim
                                    , n_steps=n_iterations, n_etypes=n_edge_types)
        self.ggc_2 = GatedGraphConv(in_feats=hid_dim, out_feats=out_dim
                                    , n_steps=n_iterations, n_etypes=n_edge_types)
        self.dropout = nn.Dropout(p=0.5)

    @staticmethod
    def ph_encoding_to_word_encoding(ph_encoding, ph2word, word_len):
        """
        ph_encoding: [batch, t_p, hid]
        ph2word: tensor [batch, t_w]
        word_len: tensor [batch]
        """
        word_encoding_for_graph, batch_word_encoding, has_word_row_idx = GraphAuxEnc._process_ph_to_word_encoding(
            ph_encoding,
            ph2word,
            word_len)
        # [batch, t_w, hid]
        return batch_word_encoding, word_encoding_for_graph

    def pad_word_encoding_to_phoneme(self, word_encoding, ph2word, t_p):
        return self._postprocess_word2ph(word_encoding, ph2word, t_p)

    @staticmethod
    def _process_ph_to_word_encoding(ph_encoding, ph2word, word_len=None):
        """
        ph_encoding: [batch, t_p, hid]
        ph2word: tensor [batch, t_w]
        word_len: tensor [batch]
        """
        word_len = word_len.reshape([-1,])
        max_len = max(word_len)
        num_nodes = sum(word_len)

        batch_word_encoding = group_hidden_by_segs(ph_encoding, ph2word, max_len)
        bs, t_p, hid = batch_word_encoding.shape
        has_word_mask = sequence_mask(word_len, max_len)  # [batch, t_p, 1]
        word_encoding = batch_word_encoding.reshape([bs * t_p, hid])
        has_word_row_idx = has_word_mask.reshape([-1])
        word_encoding = word_encoding[has_word_row_idx]
        assert word_encoding.shape[0] == num_nodes
        return word_encoding, batch_word_encoding, has_word_row_idx

    @staticmethod
    def _postprocess_word2ph(word_encoding, ph2word, t_p):
        word_encoding = F.pad(word_encoding,[0,0,1,0])
        ph2word_ = ph2word[:, :, None].repeat([1, 1, word_encoding.shape[-1]])
        out = torch.gather(word_encoding, 1, ph2word_)  # [B, T, H]
        return out

    @staticmethod
    def _repeat_one_sequence(x, d, T):
        """Repeat each frame according to duration."""
        if d.sum() == 0:
            d = d.fill_(1)
        hid = x.shape[-1]
        expanded_lst = [x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0]
        expanded = torch.cat(expanded_lst, dim=0)
        if T > expanded.shape[0]:
            expanded = torch.cat([expanded, torch.zeros([T - expanded.shape[0], hid]).to(expanded.device)], dim=0)
        return expanded

    def word_forward(self, graph_lst, word_encoding, etypes_lst):
        """
        word encoding in, word encoding out.
        """
        batched_graph = dgl.batch(graph_lst)
        inp = word_encoding
        batched_etypes = torch.cat(etypes_lst)  # [num_edges_in_batch, 1]
        assert batched_graph.num_nodes() == inp.shape[0]

        gcc1_out = self.ggc_1(batched_graph, inp, batched_etypes)
        if self.dropout_after_gae:
            gcc1_out = self.dropout(gcc1_out)
        gcc2_out = self.ggc_2(batched_graph, gcc1_out, batched_etypes)  # [num_nodes_in_batch, hin]
        if self.dropout_after_gae:
            gcc2_out = self.ggc_2(batched_graph, gcc2_out, batched_etypes)
        if self.skip_connect:
            assert self.in_dim == self.hid_dim and self.hid_dim == self.out_dim
            gcc2_out = inp + gcc1_out + gcc2_out

        word_len = torch.tensor([g.num_nodes() for g in graph_lst]).reshape([-1])
        max_len = max(word_len)
        has_word_mask = sequence_mask(word_len, max_len)  # [batch, t_p, 1]
        has_word_row_idx = has_word_mask.reshape([-1])
        bs = len(graph_lst)
        t_w = max([g.num_nodes() for g in graph_lst])
        hid = word_encoding.shape[-1]
        output = torch.zeros([bs * t_w, hid]).to(gcc2_out.device)
        output[has_word_row_idx] = gcc2_out
        output = output.reshape([bs, t_w, hid])
        word_level_output = output
        return torch.transpose(word_level_output, 1, 2)

    def forward(self, graph_lst, ph_encoding, ph2word, etypes_lst, return_word_encoding=False):
        """
        graph_lst: [list of dgl_graph]
        ph_encoding: [batch, hid, t_p]
        ph2word: [list of list[1,2,2,2,3,3,3]]
        etypes_lst: [list of etypes]; etypes: torch.LongTensor
        """
        t_p = ph_encoding.shape[-1]
        ph_encoding = ph_encoding.transpose(1,2) # [batch, t_p, hid]
        word_len = torch.tensor([g.num_nodes() for g in graph_lst]).reshape([-1])
        batched_graph = dgl.batch(graph_lst)
        inp, batched_word_encoding, has_word_row_idx = self._process_ph_to_word_encoding(ph_encoding, ph2word,
                                                                                         word_len=word_len)  # [num_nodes_in_batch, in_dim]
        bs, t_w, hid = batched_word_encoding.shape
        batched_etypes = torch.cat(etypes_lst)  # [num_edges_in_batch, 1]
        gcc1_out = self.ggc_1(batched_graph, inp, batched_etypes)
        gcc2_out = self.ggc_2(batched_graph, gcc1_out, batched_etypes)  # [num_nodes_in_batch, hin]
        # skip connection 
        gcc2_out = inp + gcc1_out + gcc2_out # [n_nodes, hid]
        
        output = torch.zeros([bs * t_w, hid]).to(gcc2_out.device)
        output[has_word_row_idx] = gcc2_out
        output = output.reshape([bs, t_w, hid])
        word_level_output = output
        output = self._postprocess_word2ph(word_level_output, ph2word, t_p)  # [batch, t_p, hid]
        output = torch.transpose(output, 1, 2)

        if return_word_encoding:
            return output, torch.transpose(word_level_output, 1, 2)
        else:
            return output

if __name__ == '__main__':
    # Unit Test for batching graphs
    from modules.syntaspeech.syntactic_graph_buider import Sentence2GraphParser, plot_dgl_sentence_graph
    parser = Sentence2GraphParser("en")

    # Unit Test for English Graph Builder
    text1 = "To be or not to be , that 's a question ."
    text2 = "I love you . You love me . Mixue ice-scream and tea ."
    graph1, etypes1 = parser.parse(text1)
    graph2, etypes2 = parser.parse(text2)
    batched_text = "<BOS> " + text1 + " <EOS>" + " " + "<BOS> " + text2 + " <EOS>"
    batched_nodes = [graph1.num_nodes(), graph2.num_nodes()]
    plot_dgl_sentence_graph(dgl.batch([graph1, graph2]), {i: w for i, w in enumerate(batched_text.split(" "))})
    etypes_lst = [etypes1, etypes2]

    # Unit Test for Graph Encoder forward
    in_feats = 4
    out_feats = 4
    enc = GraphAuxEnc(in_dim=in_feats, hid_dim=in_feats, out_dim=out_feats)
    ph2word = torch.tensor([
        [1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13, 0],
        [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ])
    inp = torch.randn([2,  in_feats, 17]) # [N_sentence, feat, ph_length]
    graph_lst = [graph1, graph2]
    out = enc(graph_lst, inp, ph2word, etypes_lst)
    print(out.shape)  # [N_sentence, feat, ph_length]
