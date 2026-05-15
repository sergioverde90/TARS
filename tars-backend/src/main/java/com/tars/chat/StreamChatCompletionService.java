package com.tars.chat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.api.dto.ChatCompletionRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
public class StreamChatCompletionService {

    private static final ExecutorService asyncPool =
            Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "sse-async");
                t.setDaemon(true);
                return t;
            });

    private static final Logger log = LoggerFactory.getLogger(StreamChatCompletionService.class);
    private static final int MAX_TOOL_ITERATIONS = 12;
    private static final Pattern REASONING_KEY = Pattern.compile("\"reasoning\"\\s*:");
    private static final Duration SSE_TIMEOUT = Duration.of(10, ChronoUnit.MINUTES);

    private final List<ToolMetadata> toolMetadata;
    private final Map<String, Function<String, String>> toolHandlers;
    private final ObjectMapper mapper = new ObjectMapper();

    public StreamChatCompletionService(
        List<ToolMetadata> toolMetadata,
        Map<String, Function<String, String>> toolHandlers
    ) {
        this.toolMetadata = toolMetadata;
        this.toolHandlers = toolHandlers;
    }

    public SseEmitter stream(ChatCompletionRequest req) {
        SseEmitter emitter = new SseEmitter(SSE_TIMEOUT.toMillis());
        List<ChatMessage> messages = convertMessages(req);
        streamRound(messages, emitter, 0, req);
        return emitter;
    }

    private void streamRound(List<ChatMessage> messages, SseEmitter emitter, int depth, ChatCompletionRequest req) {
        if (depth > MAX_TOOL_ITERATIONS) {
            log.warn("Max tool iterations reached");
            sendEvent(emitter, "[DONE]");
            emitter.complete();
            return;
        }

        log.info("Starting chat round {}, messages: {}", depth, messages.size());

        // Build request body for mlx-openai-server
        Map<String, Object> body = buildRequestBody(messages, req);
        // Read SSE stream directly from mlx-openai-server using java.net.http.HttpClient
        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMinutes(5))
            .build();

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:8080/v1/chat/completions"))
            .header("Content-Type", MediaType.APPLICATION_JSON_VALUE)
            .POST(HttpRequest.BodyPublishers.ofString(json(body)))
            .build();

        // Accumulator for tool calls found in the SSE stream
        ToolCallAccumulator accumulator = new ToolCallAccumulator();

        // Run SSE reading on a background thread to avoid blocking
        CompletableFuture.runAsync(() -> {
            try {
                InputStream is = client.send(request, HttpResponse.BodyHandlers.ofInputStream()).body();
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
                String line;
                while ((line = reader.readLine()) != null) {
                    try {
                        String data = processSseLine(line, accumulator);
                        if (data != null) {
                            sendEvent(emitter, data);
                        }
                    } catch (Exception e) {
                        log.error("Error processing SSE line", e);
                        throw e;
                    }
                }

                // After streaming, check if we need to execute tool calls
                List<ToolCall> toolCalls = accumulator.getToolCalls();

                if (!toolCalls.isEmpty()) {
                    log.info("Accumulated {} tool calls, executing...", toolCalls.size());

                    // Append the assistant message with tool calls to the conversation
                    AssistantMessage assistantMessage = AssistantMessage.from(toolCalls);
                    messages.add(assistantMessage);

                    // Execute each tool call and append results
                    for (ToolCall call : toolCalls) {
                        String result = executeToolCall(call);
                        System.out.println("result = " + result);
                        ToolResultMessage toolResult = ToolResultMessage.from(call, result);
                        System.out.println("toolResult = " + toolResult);
                        messages.add(toolResult);
                        log.info("Tool '{}' executed, result length: {}", call.name(), result.length());
                    }

                    // Recurse with the updated conversation
                    streamRound(messages, emitter, depth + 1, req);
                    return;
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
        }, asyncPool)
        .exceptionally(ex -> {
            log.error("Unhandled error in async SSE round {}", depth, ex);
            emitter.completeWithError(ex);
            return null;
        });;

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

    private String processSseLine(String line, ToolCallAccumulator accumulator) {
        if (!line.startsWith("data: ")) {
            return null;
        }
        String data = line.substring(6).trim();
        if (data.equals("[DONE]")) {
            return "[DONE]";
        }

        // Parse SSE JSON to extract tool calls
        try {
            JsonNode root = mapper.readTree(data);
            JsonNode choices = root.path("choices");
            if (choices.isArray() && !choices.isEmpty()) {
                JsonNode choice = choices.get(0);
                JsonNode finishReason = choice.path("finish_reason");
                JsonNode delta = choice.path("delta");
                JsonNode message = choice.path("message");

                // Check for tool_calls in delta (streaming chunks)
                JsonNode deltaToolCalls = delta.path("tool_calls");
                if (deltaToolCalls.isArray() && !deltaToolCalls.isEmpty()) {
                    accumulator.processDeltaToolCalls(deltaToolCalls);
                }

                // Check for tool_calls in message (final non-streaming chunk)
                JsonNode messageToolCalls = message.path("tool_calls");
                if (messageToolCalls.isArray() && !messageToolCalls.isEmpty()) {
                    accumulator.processMessageToolCalls(messageToolCalls);
                }

                // If finish_reason is "tool_calls", we've captured all tool calls
                if ("tool_calls".equals(finishReason.asText())) {
                    accumulator.markComplete();
                }
            }
        } catch (Exception e) {
            // Not JSON or parsing error, ignore for tool call processing
            log.error("Non-JSON SSE line or parse error", e);
        }

        // Rewrite "reasoning_content" -> "reasoning" to match what the client expects
        return REASONING_KEY.matcher(data).replaceAll("\"reasoning\":");
    }

    private Map<String, Object> buildRequestBody(List<ChatMessage> messages, ChatCompletionRequest req) {
        List<Map<String, Object>> apiMessages = messages.stream()
            .map(m -> {
                switch (m) {
                    case SystemMessage sys -> {
                        return Map.<String, Object>of("role", "system", "content", sys.text());
                    }
                    case UserMessage user -> {
                        return Map.<String, Object>of("role", "user", "content", user.text());
                    }
                    case AssistantMessage ai -> {
                        Map<String, Object> msg = new HashMap<>();
                        msg.put("role", "assistant");
                        msg.put("content", ai.text() != null ? ai.text() : "");
                        // ✅ Include tool_calls so the model remembers it made them
                        if (ai.toolCalls() != null && !ai.toolCalls().isEmpty()) {
                            List<Map<String, Object>> tcList = ai.toolCalls().stream()
                                .map(tc -> {
                                    Map<String, Object> tcMap = new HashMap<>();
                                    tcMap.put("id", tc.id());
                                    tcMap.put("type", "function");
                                    tcMap.put("function", Map.of(
                                            "name", tc.name(),
                                            "arguments", tc.arguments()
                                    ));
                                    return tcMap;
                                }).toList();
                            msg.put("tool_calls", tcList);
                        }
                        return msg;
                    }
                    case ToolResultMessage tool -> {
                        // ✅ tool_call_id must match the id in the assistant message above
                        return Map.<String, Object>of(
                            "role", "tool",
                            "tool_call_id", tool.toolCallId(),
                            "content", tool.result()
                        );
                    }
                    default -> {
                        return Map.<String, Object>of("role", "user", "content", String.valueOf(m));
                    }
                }
            })
            .toList();

        // Build tools array from ToolMetadata
        List<Map<String, Object>> tools = toolMetadata.stream()
            .map(spec -> {
                Map<String, Object> tool = new HashMap<>();
                tool.put("type", "function");
                Map<String, Object> function = new HashMap<>();
                function.put("name", spec.name());
                function.put("description", spec.description());
                function.put("parameters", spec.parameters());
                tool.put("function", function);
                return tool;
            })
            .toList();

        Map<String, Object> body = new HashMap<>();
        body.put("messages", apiMessages);
        body.put("tools", tools);
        body.put("tool_choice", "auto");
        body.put("stream", true);
        body.put("temperature", 0.8);
        body.put("max_tokens", 8192);
        body.put("min_p", 0.06);
        body.put("presence_penalty", 1.2);
        body.put("repeat_penalty", 1.05);
        
        // Use enable_thinking from request if provided, default to true
        boolean enableThinking = true;
        if (req.chatTemplateKwargs() != null) {
            String thinkingValue = req.chatTemplateKwargs().getOrDefault("enable_thinking", "true");
            enableThinking = Boolean.parseBoolean(thinkingValue);
        }
        body.put("chat_template_kwargs", Map.of("enable_thinking", enableThinking));
        return body;
    }

    private String executeToolCall(ToolCall call) {
        Function<String, String> handler = toolHandlers.get(call.name());
        if (handler == null) {
            log.warn("Unknown tool: {}", call.name());
            return "Unknown tool: " + call.name();
        }
        try {
            return handler.apply(call.arguments());
        } catch (Exception e) {
            log.error("Tool '{}' failed", call.name(), e);
            return "Tool error: " + e.getMessage();
        }
    }

    private List<ChatMessage> convertMessages(ChatCompletionRequest req) {
        Map<String, String> stringStringMap = req.chatTemplateKwargs();
        List<ChatCompletionRequest.Message> messages = req.messages();
        return new ArrayList<>(messages.stream()
            .map(m -> switch (m.role()) {
                case "system"    -> SystemMessage.from(extractText(m.content()) + TOOL_INSTRUCTION);
                case "assistant" -> AssistantMessage.from(extractText(m.content()));
                default          -> UserMessage.from(extractText(m.content()), stringStringMap);
            })
            .toList());
    }

    private String extractText(List<ChatCompletionRequest.ContentItem> content) {
        if (content == null) {
            return "";
        }
        return content.stream()
            .filter(item -> "text".equals(item.type()))
            .map(ChatCompletionRequest.ContentItem::text)
            .filter(Objects::nonNull)
            .collect(Collectors.joining("\n"));
    }

    private void sendEvent(SseEmitter emitter, String data) {
        try {
            emitter.send(SseEmitter.event().data(data, MediaType.TEXT_PLAIN));
        } catch (IOException e) {
            log.error("Failed to send SSE event", e);
            throw new UncheckedIOException(e);
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
