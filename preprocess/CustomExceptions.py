from Encoder import Encoder

class IncompatibleEncoder(Exception):
    """Exception resulting from trying to perform an operation on an encoder that is not the correct type
    """

    def __init__(self, required_encoder_type,encoder:Encoder):

        self.message = "Operation requires encoder of type '{}', but encoder is of type '{}'".format(required_encoder_type,encoder.encoder_type)
        super().__init__(self.message)