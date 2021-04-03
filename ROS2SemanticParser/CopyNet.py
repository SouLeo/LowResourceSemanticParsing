import torch
from allennlp.models.archival import load_archive
from allennlp_models.generation.predictors.seq2seq import Seq2SeqPredictor
from allennlp_models.generation import CopyNetSeq2Seq
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.attention.bilinear_attention import BilinearAttention
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp_models.generation.dataset_readers.copynet_seq2seq import CopyNetDatasetReader


class SemanticParser:
    def __init__(self):
        # CopyNet model initialization parameters
        self.vocabulary = Vocabulary()
        self.vocabulary = self.vocabulary.from_files("C:/Users/Selma/PycharmProjects/ROS2SemanticParser/"
                                                     "CN_model_weights/no_embedds/model.tar.gz")
        self.source_embedder = BasicTextFieldEmbedder(token_embedders={'tokens': Embedding(
            num_embeddings=self.vocabulary.get_vocab_size('source_tokens'), embedding_dim=310)})
        self.dataset_reader = CopyNetDatasetReader(target_namespace="target_tokens")
        self.encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=310, hidden_size=128, num_layers=1,
                                                           batch_first=True))
        self.attention = BilinearAttention(vector_dim=128, matrix_dim=128)
        self.beam_size = 5
        self.max_decoding_steps = 200
        self.target_embedding_dim = 150

        self.semantic_parser = CopyNetSeq2Seq(vocab=self.vocabulary, source_embedder=self.source_embedder,
                                              encoder=self.encoder, attention=self.attention, beam_size=self.beam_size,
                                              max_decoding_steps=self.max_decoding_steps,
                                              target_embedding_dim=self.target_embedding_dim)

    def load_existing_model(self):
        # load previously trained model weights
        with open("C:/Users/Selma/PycharmProjects/ROS2SemanticParser/CN_model_weights/no_embedds/"
                  "best.th",
                  'rb') as f:
            self.semantic_parser.load_state_dict(torch.load(f))

    def training(self):
        raise NotImplemented
        # training

    def predict_from_file(self):
        raise NotImplemented
        # prediction from tsv file

    def predict_continuous(self, nl_input):
        # predictor = Seq2SeqPredictor(self.semantic_parser, self.dataset_reader)
        archive = load_archive("C:/Users/Selma/PycharmProjects/ROS2SemanticParser/"
                                                     "CN_model_weights/no_embedds/model.tar.gz")
        predictor = Seq2SeqPredictor.from_archive(archive)
        output = predictor.predict(nl_input)
        print(output['predicted_tokens'][0])