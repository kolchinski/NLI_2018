from basemodel import BaseModel
import seq2seq_train_pytorch


class Seq2SeqPytorch(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self)
        self.lr = args.lr
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.batches_per_epoch = args.batches_per_epoch
        self.test_batches_per_epoch = args.test_batches_per_epoch
        self.cuda = cuda

        self.encoder =
        self.decoder =
        self.encoder_optimizer =
        self.decoder_optimizer =
        self.criterion =

    def train(self, train_data):
        train_loss = seq2seq_train_pytorch.train(
            input_variable=input_variable,
            target_variable=target_variable,
            encoder=self.encoder,
            decoder=self.decoder,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            criterion=self.criterion,
            max_length=self.max_length,
        )

        return train_loss

    def eval(self, eval_data):
