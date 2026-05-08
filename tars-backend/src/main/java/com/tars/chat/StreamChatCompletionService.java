package com.tars.chat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.api.dto.ChatCompletionChunk;
import com.tars.api.dto.ChatCompletionRequest;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.regex.Pattern;

@Service
public class StreamChatCompletionService {

    private static final Logger log = LoggerFactory.getLogger(StreamChatCompletionService.class);
    private static final int MAX_TOOL_ITERATIONS = 12;
    private static final long SSE_TIMEOUT = 300_000L;
    private static final Pattern REASONING_KEY = Pattern.compile("\"reasoning\"\\s*:");

    private final List<ToolSpecification> toolSpecs;
    private final Map<String, Function<String, String>> toolHandlers;
    private final ObjectMapper mapper = new ObjectMapper();

    public StreamChatCompletionService(
        List<ToolSpecification> toolSpecifications,
        Map<String, Function<String, String>> toolHandlers
    ) {
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

        // Build request body for mlx-openai-server
        Map<String, Object> body = buildRequestBody(messages);

        // Read SSE stream directly from mlx-openai-server using java.net.http.HttpClient
        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMinutes(5))
            .build();

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:8080/v1/chat/completions"))
            .header("Content-Type", MediaType.APPLICATION_JSON_VALUE)
            .POST(HttpRequest.BodyPublishers.ofString(json(body)))
            .build();

        // Run SSE reading on a background thread to avoid blocking
        CompletableFuture.runAsync(() -> {
            try {
                InputStream is = client.send(request, HttpResponse.BodyHandlers.ofInputStream()).body();
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
                String line;
                while ((line = reader.readLine()) != null) {
                    try {
                        String data = processSseLine(line);
                        if (data != null) {
                            sendEvent(emitter, data);
                        }
                    } catch (Exception e) {
                        log.error("Error processing SSE line", e);
                    }
                }

                // After streaming, send [DONE]
                sendEvent(emitter, "[DONE]");
                emitter.complete();
            } catch (IOException | InterruptedException e) {
                if (e instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                log.error("SSE streaming error", e);
                emitter.completeWithError(e);
            }
        });

        // Handle emitter timeout/cancellation
        emitter.onCompletion(() -> {
            // Stream completed successfully
        });
        emitter.onTimeout(() -> {
            log.warn("SSE timeout");
            sendEvent(emitter, "[DONE]");
            emitter.complete();
        });
        emitter.onError(e -> {
            log.error("SSE error", e);
        });
    }

    private String processSseLine(String line) {
        if (!line.startsWith("data: ")) {
            return null;
        }
        String data = line.substring(6).trim();
        if (data.equals("[DONE]")) {
            return "[DONE]";
        }
        // Rewrite "reasoning_content" -> "reasoning" to match what the client expects
        return REASONING_KEY.matcher(data).replaceAll("\"reasoning\":");
    }

    private Map<String, Object> buildRequestBody(List<ChatMessage> messages) {
        List<Map<String, String>> apiMessages = messages.stream()
            .map(m -> {
                String role;
                String content;
                if (m instanceof SystemMessage sys) {
                    role = "system";
                    content = sys.text();
                } else if (m instanceof UserMessage user) {
                    role = "user";
                    content = user.singleText();
                } else if (m instanceof AiMessage ai) {
                    role = "assistant";
                    content = ai.text();
                } else if (m instanceof ToolExecutionResultMessage tool) {
                    role = "tool";
                    content = tool.text();
                } else {
                    role = "user";
                    content = String.valueOf(m);
                }
                return Map.of("role", role, "content", content);
            })
            .toList();

        return Map.of(
            "model", "mlx-community/gemma-4-26b-a4b-it-4bit",
            "messages", apiMessages,
            "stream", true,
            "temperature", 0.8,
            "max_tokens", 131072,
            "min_p", 0.06,
            "presence_penalty", 1.2,
            "repeat_penalty", 1.05,
            "chat_template_kwargs", Map.of("enable_thinking", true)
        );
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

    private List<ChatMessage> convertMessages(List<ChatCompletionRequest.Message> messages) {
        return messages.stream()
            .map(m -> switch (m.role()) {
                case "system"    -> (ChatMessage) SystemMessage.from(m.content() + TOOL_INSTRUCTION);
                case "assistant" -> AiMessage.from((String) m.content());
                default          -> UserMessage.from((String) m.content());
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

    private static final String TOOL_INSTRUCTION =
        "\nFor real-time queries (date, time, routes, current events) ALWAYS call the appropriate tool — never guess.";

    private String json(Map<String, Object> map) {
        try {
            return mapper.writeValueAsString(map);
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize JSON", e);
        }
    }
}
