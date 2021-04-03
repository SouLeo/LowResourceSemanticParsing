from allennlp_models.generation.models import copynet_seq2seq
from allennlp_models.generation.dataset_readers import copynet_seq2seq as data_reader
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder as embedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.attention import bilinear_attention
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.training.optimizers import AdamOptimizer
import DataInput
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DataInput.createTrainLoader()
    UMRF_dataset_reader = data_reader.CopyNetDatasetReader(target_namespace='target_side')
    dataset = UMRF_dataset_reader.read('UMRF_train.tsv')
    vocab = Vocabulary.from_instances(dataset)
    src = vocab.get_index_to_token_vocabulary()
    targ = vocab.get_index_to_token_vocabulary(namespace='target_side')

    with open('src_voc.txt', 'w') as file:
        for key, value in src.items():
                file.write('{} {}'.format(value, key) + '\n')

    with open('targ_voc.txt', 'w') as file:
        for key, value in targ.items():
                file.write('{} {}'.format(value, key) + '\n')
    # print('hi')
    # source_embed = embedder({"tokens": Embedding(embedding_dim=256, num_embeddings=310)})
    # attn = bilinear_attention.BilinearAttention(vector_dim=310, matrix_dim=1500)
    # encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(310, 1500, batch_first=True))
    # UMRF_CopyNet_model = copynet_seq2seq.CopyNetSeq2Seq(vocab=vocab, source_embedder=source_embed,
    #                                                     encoder=encoder, attention=attn,
    #                                                     beam_size=5, max_decoding_steps=50)
    #
    #
    # optimizer = AdamOptimizer()
    # trainer = Trainer()
