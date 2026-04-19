package com.tars.chat;

import com.tars.api.dto.ChatCompletionRequest;
import com.tars.api.dto.ChatCompletionResponse;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Service
public class ChatCompletionService {

    private static final Logger log = LoggerFactory.getLogger(ChatCompletionService.class);
    private static final int MAX_TOOL_ITERATIONS = 5;

    private final ChatModel chatModel;
    private final List<ToolSpecification> toolSpecs;
    private final Map<String, Function<String, String>> toolHandlers;

    public ChatCompletionService(
        ChatModel chatModel,
        List<ToolSpecification> toolSpecifications,
        Map<String, Function<String, String>> toolHandlers
    ) {
        this.chatModel = chatModel;
        this.toolSpecs = toolSpecifications;
        this.toolHandlers = toolHandlers;
    }

    public ChatCompletionResponse chat(ChatCompletionRequest req) {
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
}
