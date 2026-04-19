package com.tars.chat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.api.dto.ChatCompletionChunk;
import com.tars.api.dto.ChatCompletionRequest;
import com.tars.api.dto.ChatCompletionResponse;
import com.tars.tools.DistanceTool;
import com.tars.tools.SearchTool;
import com.tars.tools.TimeTool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Service
public class TarsChatService {

    private static final Logger log = LoggerFactory.getLogger(TarsChatService.class);
    private static final int MAX_TOOL_ITERATIONS = 5;

    private final ChatModel chatModel;
    private final StreamingChatModel streamingModel;
    private final List<ToolSpecification> toolSpecs;
    private final Map<String, Function<String, String>> toolHandlers;
    private final ObjectMapper mapper = new ObjectMapper();

    public TarsChatService(
        ChatModel chatModel,
        StreamingChatModel streamingModel,
        TimeTool timeTool,
        DistanceTool distanceTool,
        SearchTool searchTool
    ) {
        this.chatModel = chatModel;
        this.streamingModel = streamingModel;

        // toolSpecificationsFrom takes a single Object in 1.0.0-rc1
        this.toolSpecs = new ArrayList<>();
        this.toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(timeTool));
        this.toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(distanceTool));
        this.toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(searchTool));

        log.info("Registered tool specs: {}", this.toolSpecs.stream().map(ToolSpecification::name).toList());

        this.toolHandlers = new HashMap<>();
        this.toolHandlers.put("currentTime", args -> timeTool.currentTime());
        this.toolHandlers.put("drivingDistance", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return distanceTool.drivingDistance(
                    node.path("from").asText(),
                    node.path("to").asText()
                );
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });
        this.toolHandlers.put("webSearch", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return searchTool.webSearch(node.path("query").asText());
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });
    }

    public Mono<ChatCompletionResponse> chat(ChatCompletionRequest req) {
        return Mono.fromCallable(() -> {
            List<ChatMessage> messages = new ArrayList<>(convertMessages(req.messages()));

            for (int i = 0; i <= MAX_TOOL_ITERATIONS; i++) {
                ChatRequest request = ChatRequest.builder()
                    .messages(messages)
                    .toolSpecifications(toolSpecs)
                    .build();

                ChatResponse response = chatModel.chat(request);
                AiMessage aiMessage = response.aiMessage();

                if (!aiMessage.hasToolExecutionRequests()) {
                    return ChatCompletionResponse.of(aiMessage.text());
                }

                log.info("Tool calls: {}", aiMessage.toolExecutionRequests().stream()
                    .map(ToolExecutionRequest::name).toList());

                messages.add(aiMessage);
                for (ToolExecutionRequest toolReq : aiMessage.toolExecutionRequests()) {
                    String result = executeToolCall(toolReq);
                    log.info("Tool '{}' → {}", toolReq.name(), result);
                    messages.add(ToolExecutionResultMessage.from(toolReq, result));
                }
            }

            log.warn("Max tool iterations reached");
            return ChatCompletionResponse.of("I ran out of tool iterations. Try again.");
        }).subscribeOn(Schedulers.boundedElastic());
    }

    public Flux<ServerSentEvent<String>> stream(ChatCompletionRequest req) {
        List<ChatMessage> messages = convertMessages(req.messages());
        Sinks.Many<ServerSentEvent<String>> sink = Sinks.many().unicast().onBackpressureBuffer();
        streamRound(messages, sink, 0);
        return sink.asFlux();
    }

    private void streamRound(List<ChatMessage> messages, Sinks.Many<ServerSentEvent<String>> sink, int depth) {
        if (depth > MAX_TOOL_ITERATIONS) {
            log.warn("Max tool iterations reached");
            sink.tryEmitNext(sse("[DONE]"));
            sink.tryEmitComplete();
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
                    sink.tryEmitNext(sse(json));
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

                    streamRound(updated, sink, depth + 1);
                } else {
                    try {
                        String json = mapper.writeValueAsString(ChatCompletionChunk.finish());
                        sink.tryEmitNext(sse(json));
                        sink.tryEmitNext(sse("[DONE]"));
                        sink.tryEmitComplete();
                    } catch (Exception e) {
                        sink.tryEmitError(e);
                    }
                }
            }

            @Override
            public void onError(Throwable error) {
                log.error("LLM error", error);
                sink.tryEmitError(error);
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

    private static ServerSentEvent<String> sse(String data) {
        return ServerSentEvent.builder(data).build();
    }
}
