class IncompatibleEncoder(Exception):
    """Exception resulting from trying to perform an operation on an encoder that is not the correct type
    """

    def __init__(self, required_encoder_type,actual_encoder_type):

        self.message = "Operation requires encoder of type '{}', but encoder is of type '{}'".format(required_encoder_type,actual_encoder_type)
        super().__init__(self.message)

class FilteringNotEnabled(Exception):
    """Exception resulting from apply a filter to data when no filter has been configured for the given encoder
    """

    def __init__(self,actual_filter_enabled_status):

        self.message = "Attempted to apply a filter to data, but filter enabled status was '{}'".format(actual_filter_enabled_status)
        super().__init__(self.message)