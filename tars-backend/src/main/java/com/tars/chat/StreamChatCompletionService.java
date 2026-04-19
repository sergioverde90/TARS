package com.tars.chat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.api.dto.ChatCompletionChunk;
import com.tars.api.dto.ChatCompletionRequest;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Service
public class StreamChatCompletionService {

    private static final Logger log = LoggerFactory.getLogger(StreamChatCompletionService.class);
    private static final int MAX_TOOL_ITERATIONS = 5;
    private static final long SSE_TIMEOUT = 300_000L;

    private final StreamingChatModel streamingModel;
    private final List<ToolSpecification> toolSpecs;
    private final Map<String, Function<String, String>> toolHandlers;
    private final ObjectMapper mapper = new ObjectMapper();

    public StreamChatCompletionService(
        StreamingChatModel streamingModel,
        List<ToolSpecification> toolSpecifications,
        Map<String, Function<String, String>> toolHandlers
    ) {
        this.streamingModel = streamingModel;
        this.toolSpecs = toolSpecifications;
        this.toolHandlers = toolHandlers;
    }

    public SseEmitter stream(ChatCompletionRequest req) {
        SseEmitter emitter = new SseEmitter(SSE_TIMEOUT);
        List<ChatMessage> messages = convertMessages(req.messages());
        streamRound(messages, emitter, 0);
        return emitter;
    }

    private void streamRound(List<ChatMessage> messages, SseEmitter emitter, int depth) {
        if (depth > MAX_TOOL_ITERATIONS) {
            log.warn("Max tool iterations reached");
            sendEvent(emitter, "[DONE]");
            emitter.complete();
            return;
        }

        log.info("Starting chat round {}, messages: {}", depth, messages.size());

        ChatRequest request = ChatRequest.builder()
            .messages(messages)
            .toolSpecifications(toolSpecs)
            .build();

        streamingModel.chat(request, new StreamingChatResponseHandler() {

            @Override
            public void onPartialResponse(String token) {
                try {
                    String json = mapper.writeValueAsString(ChatCompletionChunk.contentDelta(token));
                    sendEvent(emitter, json);
                } catch (Exception e) {
                    log.error("Serialization error", e);
                }
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                AiMessage aiMessage = response.aiMessage();

                log.info("LLM response complete: {}", aiMessage);
                log.info("LLM response tool calls: {}", aiMessage.toolExecutionRequests());

                if (aiMessage.hasToolExecutionRequests()) {
                    List<ToolExecutionRequest> toolRequests = aiMessage.toolExecutionRequests();
                    log.info("Tool calls: {}", toolRequests.stream().map(ToolExecutionRequest::name).toList());

                    List<ChatMessage> updated = new ArrayList<>(messages);
                    updated.add(aiMessage);

                    for (ToolExecutionRequest req : toolRequests) {
                        String result = executeToolCall(req);
                        log.info("Tool '{}' → {}", req.name(), result);
                        updated.add(ToolExecutionResultMessage.from(req, result));
                    }

                    streamRound(updated, emitter, depth + 1);
                } else {
                    try {
                        String json = mapper.writeValueAsString(ChatCompletionChunk.finish());
                        sendEvent(emitter, json);
                        sendEvent(emitter, "[DONE]");
                        emitter.complete();
                    } catch (Exception e) {
                        emitter.completeWithError(e);
                    }
                }
            }

            @Override
            public void onError(Throwable error) {
                log.error("LLM error", error);
                emitter.completeWithError(error);
            }
        });
    }

    private String executeToolCall(ToolExecutionRequest req) {
        Function<String, String> handler = toolHandlers.get(req.name());
        if (handler == null) {
            log.warn("Unknown tool: {}", req.name());
            return "Unknown tool: " + req.name();
        }
        try {
            return handler.apply(req.arguments());
        } catch (Exception e) {
            log.error("Tool '{}' failed", req.name(), e);
            return "Tool error: " + e.getMessage();
        }
    }

    private static final String TOOL_INSTRUCTION =
        "\nFor real-time queries (date, time, routes, current events) ALWAYS call the appropriate tool — never guess.";

    private List<ChatMessage> convertMessages(List<ChatCompletionRequest.Message> messages) {
        return messages.stream()
            .map(m -> switch (m.role()) {
                case "system"    -> (ChatMessage) SystemMessage.from(m.content() + TOOL_INSTRUCTION);
                case "assistant" -> AiMessage.from(m.content());
                default          -> UserMessage.from(m.content());
            })
            .toList();
    }

    private void sendEvent(SseEmitter emitter, String data) {
        try {
            emitter.send(SseEmitter.event().data(data, MediaType.TEXT_PLAIN));
        } catch (IOException e) {
            log.error("Failed to send SSE event", e);
        }
    }
}
