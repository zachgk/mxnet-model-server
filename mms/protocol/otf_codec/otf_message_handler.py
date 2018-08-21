import struct
from mms.utils.validators.model_worker_message_validator import ModelWorkerOtfMessageValidator as validate
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes

int_size = 4
END_OF_LIST = -2
START_OF_LIST = -1


class OtfCodecHandler:
    def __retrieve_load_msg(self, data):
        """
        MSG Frame Format:
        "
        | 1.0 | int cmd_length | cmd value | int model-name length | model-name value |
        | int model-path length | model-path value |
        | int batch-size length | batch-size value | int handler length | handler value |
        | int gpu id length | gpu ID value |
        "
        :param data:
        :return:
        """
        msg = dict()
        offset = 0
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size
        msg['modelName'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size
        msg['modelPath'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length
        msg['batchSize'] = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size
        length = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size
        msg['handler'] = struct.unpack('!{}s'.format(length), data[offset:offset+length])[0].decode()
        offset += length
        gpu_id = struct.unpack('!i', data[offset: offset+int_size])[0]
        if gpu_id > 0:
            msg['gpu'] = gpu_id

        return "load", msg

    def __retrieve_model_inputs(self, data, msg, content_type):
        offset = 0
        end = False
        while end is False:
            model_input = dict()
            length = struct.unpack('!i', data[offset:offset+int_size])[0]
            offset += int_size
            if length > 0:
                model_input['name'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length
            elif length == END_OF_LIST:
                end = True
                continue

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size

            if length > 0:
                model_input['contentType'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size

            if length > 0:
                if ("contentType" in model_input and "json" in model_input['contentType'].lower()) or \
                   ("json" in content_type.lower()):
                    model_input['value'] = struct.unpack('!{}s'.format(length), data[offset:offset+length])[0].decode()
                elif ("contentType" in model_input and "jpeg" in model_input['contentType'].lower()) or \
                     ("jpeg" in content_type.lower()):
                    model_input['value'] = data[offset: offset+length]
                else:
                    raise MMSError(ModelServerErrorCodes.UNKNOWN_CONTENT_TYPE, "Unknown contentType given for the data")
                offset += length
            msg.append(model_input)
        return offset

    def __retrieve_request_batch(self, data, msg):
        offset = 0
        end = False
        while end is False:
            reqBatch = dict()
            length = struct.unpack('!i', data[offset:offset+int_size])[0]
            offset += int_size
            if length > 0:
                reqBatch['requestId'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length
            elif length == END_OF_LIST:
                end = True
                continue

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size
            if length > 0:
                reqBatch['contentType'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size
            if length == START_OF_LIST:  # Beginning of list
                reqBatch['modelInputs'] = list()
                offset += self.__retrieve_model_inputs(data[offset:], reqBatch['modelInputs'], reqBatch['contentType'])

            msg.append(reqBatch)

    def __retrieve_inference_msg(self, data):
        msg = dict()
        offset = 0
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size

        if length > 0:
            msg['modelName'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length

        length = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size

        if length == START_OF_LIST:
            msg['requestBatch'] = list()
            self.__retrieve_request_batch(data[offset:], msg['requestBatch'])
        return "predict", msg

    def retrieve_msg(self, data):
        # Validate its beginning of a message
        if validate.validate_message(data=data) is False:
            return MMSError(ModelServerErrorCodes.INVALID_MESSAGE, "Invalid message received")

        cmd = struct.unpack('!i', data[8:12])[0]

        if cmd == 0x01:
            return self.__retrieve_load_msg(data[12:])
        elif cmd == 0x02:
            return self.__retrieve_inference_msg(data[12:])
        else:
            return "unknown", "Wrong command "

    # def __encode_response(self, **kwargs):
    #     try:
    #         req_id_map = kwargs['req_id_map']
    #         invalid_reqs = kwargs['invalid_reqs']
    #         ret = kwargs['resp']
    #         msg = bytearray()
    #         for idx, val in enumerate(ret):
    #             msg[0] =
    #             result.update({"requestId": req_id_map[idx]})
    #             result.update({"code": 200})
    #
    #             if isinstance(val, str):
    #                 value = ModelWorkerCodecHelper.encode_msg(encoding, val.encode('utf-8'))
    #             elif isinstance(val, bytes):
    #                 value = ModelWorkerCodecHelper.encode_msg(encoding, val)
    #             else:
    #                 value = ModelWorkerCodecHelper.encode_msg(encoding, json.dumps(val).encode('utf-8'))
    #
    #             result.update({"value": value})
    #             result.update({"encoding": encoding})
    #
    #         for req in invalid_reqs.keys():
    #             result.update({"requestId": req})
    #             result.update({"code": invalid_reqs.get(req)})
    #             result.update({"value": ModelWorkerCodecHelper.encode_msg(encoding,
    #                                                                       "Invalid input provided".encode('utf-8'))})
    #             result.update({"encoding": encoding})
    #
    #         resp = [result]
    #
    #     except Exception:
    #         # TODO: Return a invalid response
    #         pass
    #
    # def create_response(self, cmd, **kwargs):
    #     if cmd == 2: # Predict request response
    #         return self.encode_response(kwargs)
