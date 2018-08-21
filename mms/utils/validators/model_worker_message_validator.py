import struct


class ModelWorkerOtfMessageValidator(object):
    @staticmethod
    def validate_message(data):
        if struct.unpack('!d', data[0:8])[0] == 1.0:
            return True
        return False

    @staticmethod
    def validate_load_message(data):
        return True
