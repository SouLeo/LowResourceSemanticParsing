from DataInput import *
from Seq2SeqA import *


class Seq2SeqSemanticParser(object):
    def __init__(self, model, output_indexer, criterion):
        self.model = model
        self.output_indexer = output_indexer
        self.criterion = criterion

    def decode(self, test_loader, sos_seq):
        self.model.eval()
        with torch.no_grad():
            for sents, labels in test_loader:
                output = self.model(sents, labels, sos_seq)

                out_q = output.argmax(2)
                w, h = out_q.shape[0], out_q.shape[1]
                decoded_out = [[self.output_indexer.get_object(int(out_q[x, y])) for x in range(w)] for y in range(h)]
        return decoded_out


def train_model_encdec(train_loader, output_indexer, sos_seq):
    encoder = EncoderRNN(input_size=238, hidden_size=768)
    decoder = AttnDecoderRNN(hidden_size=768, output_size=255)
    model = Seq2Seq(encoder, decoder)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    model.optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    print(model)
    epochs = 5  # TODO: Change back to 10
    model.train()
    for epoch in range(0, epochs):
        print('epoch num: ')
        print(epoch)
        epoch_loss = 0
        for sents, labels in train_loader:

            model.optimizer.zero_grad()
            output = model.forward(sents, labels, sos_seq)

            output = output[1:].view(-1, output.shape[-1])
            labels_t = torch.transpose(labels, 0, 1)
            labels = torch.reshape(labels_t[1:], (-1,))

            labels_long = labels.type(torch.LongTensor)
            loss = criterion(output, labels_long)
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            model.optimizer.step()

        loss_normalized = epoch_loss
        print(loss_normalized)
    return Seq2SeqSemanticParser(model, output_indexer, criterion)


if __name__ == '__main__':
    train_loader, output_indexer, sos_seq, test_loader = createTrainLoader()
    decoder = train_model_encdec(train_loader, output_indexer, sos_seq)

    decodings = decoder.decode(test_loader, sos_seq)
    outfile="decoding_outputs_UMRF.txt"
    with open(outfile, "w") as out:
        for sent in decodings:
            for token in sent:
                out.write(str(token) + ' ')
            out.write('\n')
    out.close()

# NOTES TO MYSELF:
# 1) INCREASE THE TRAINING DATA TO FULL SET (MULTIPLE OF BATCHSIZE)
# 2) ADJUST SOS_SEQUENCE LENGTH TO BATCH SIZE IN THE DATAINPUT FILE
# 3) ADJUST LINE 152 IN SEQ2SEQ FOR SOS_SEQUENCE TO REPEAT LEN(TRAINING DATA)/BATCH_SIZE
# 4) INCREASE EPOCHS TO 10
