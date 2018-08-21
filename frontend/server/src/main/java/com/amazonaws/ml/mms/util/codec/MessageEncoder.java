/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.util.codec;

import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelInputs;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.RequestBatch;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;


@ChannelHandler.Sharable
public class MessageEncoder extends MessageToByteEncoder<BaseModelRequest> {

    private void encodeRequestBatch(RequestBatch req, ByteBuf out) {
        out.writeInt(req.getRequestId().length());
        out.writeBytes(req.getRequestId().getBytes());

        if(req.getContentType() != null) {
            out.writeInt(req.getContentType().length());
            out.writeBytes(req.getContentType().getBytes());
        } else {
            out.writeInt(0);// Length 0
        }

        out.writeInt(-1); // Start of List
        for(ModelInputs input: req.getModelInputs()) {
            encodeModelInputs(input, out);
        }
        out.writeInt(-2); // End of List
    }

    private void encodeModelInputs(ModelInputs modelInputs, ByteBuf out) {
        out.writeInt(modelInputs.getName().length());
        out.writeBytes(modelInputs.getName().getBytes());

        if(modelInputs.getContentType() != null) {
            out.writeInt(modelInputs.getContentType().length());
            out.writeBytes(modelInputs.getContentType().getBytes());
        } else {
            out.writeInt(0); // Length 0
        }

        out.writeInt(modelInputs.getValue().length);
        out.writeBytes(modelInputs.getValue());
    }

    @Override
    protected void encode(ChannelHandlerContext ctx, BaseModelRequest msg, ByteBuf out) throws Exception {
        if (msg instanceof ModelLoadModelRequest) {
            out.writeDouble(1.0); // SOM
            out.writeInt(1); // load 1
            out.writeInt(msg.getModelName().length());
            out.writeBytes(msg.getModelName().getBytes());

            out.writeInt(((ModelLoadModelRequest) msg).getModelPath().length());
            out.writeBytes(((ModelLoadModelRequest) msg).getModelPath().getBytes());

            if (((ModelLoadModelRequest) msg).getBatchSize() >= 0) {
                out.writeInt(((ModelLoadModelRequest) msg).getBatchSize());
            } else {
                out.writeInt(1);
            }

            out.writeInt(((ModelLoadModelRequest) msg).getHandler().length());
            out.writeBytes(((ModelLoadModelRequest) msg).getHandler().getBytes());

            if (((ModelLoadModelRequest) msg).getGpu() != null) {
                out.writeInt(Integer.getInteger(((ModelLoadModelRequest) msg).getGpu()));
            } else {
                out.writeInt(-1);
            }
        } else if (msg instanceof ModelInferenceRequest) {
            out.writeDouble(1.0);
            out.writeInt(2); // Predict/inference: 2
            out.writeInt(msg.getModelName().length());
            out.writeBytes(msg.getModelName().getBytes());

            out.writeInt(-1); // Start of List
            for (RequestBatch batch: ((ModelInferenceRequest) msg).getRequestBatch()) {
                encodeRequestBatch(batch, out);
            }
            out.writeInt(-2); // End of List
        }
        out.writeBytes("\r\n".getBytes()); // EOM
    }
}
