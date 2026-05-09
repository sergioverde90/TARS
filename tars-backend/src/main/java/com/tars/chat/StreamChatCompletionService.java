package com.tars.chat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.api.dto.ChatCompletionRequest;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.regex.Pattern;

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
                    }
                }

                // After streaming, check if we need to execute tool calls
                List<ToolExecutionRequest> toolCalls = accumulator.getToolCalls();

                System.out.println("toolCalls = " + toolCalls);

                if (!toolCalls.isEmpty()) {
                    log.info("Accumulated {} tool calls, executing...", toolCalls.size());

                    // Append the assistant message with tool calls to the conversation
                    AiMessage assistantMessage = AiMessage.from(toolCalls);
                    messages.add(assistantMessage);

                    // Execute each tool call and append results
                    for (ToolExecutionRequest req : toolCalls) {
                        String result = executeToolCall(req);
                        System.out.println("result = " + result);
                        ToolExecutionResultMessage toolResult = ToolExecutionResultMessage.from(req, result);
                        messages.add(toolResult);
                        log.info("Tool '{}' executed, result length: {}", req.name(), result.length());
                    }

                    // Recurse with the updated conversation
                    streamRound(messages, emitter, depth + 1);
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
            if (choices.isArray() && choices.size() > 0) {
                JsonNode choice = choices.get(0);
                JsonNode finishReason = choice.path("finish_reason");
                JsonNode delta = choice.path("delta");
                JsonNode message = choice.path("message");

                // Check for tool_calls in delta (streaming chunks)
                JsonNode deltaToolCalls = delta.path("tool_calls");
                if (deltaToolCalls.isArray() && deltaToolCalls.size() > 0) {
                    accumulator.processDeltaToolCalls(deltaToolCalls);
                }

                // Check for tool_calls in message (final non-streaming chunk)
                JsonNode messageToolCalls = message.path("tool_calls");
                if (messageToolCalls.isArray() && messageToolCalls.size() > 0) {
                    accumulator.processMessageToolCalls(messageToolCalls);
                }

                // If finish_reason is "tool_calls", we've captured all tool calls
                if ("tool_calls".equals(finishReason.asText())) {
                    accumulator.markComplete();
                }
            }
        } catch (Exception e) {
            // Not JSON or parsing error, ignore for tool call processing
            log.debug("Non-JSON SSE line or parse error", e);
        }

        // Rewrite "reasoning_content" -> "reasoning" to match what the client expects
        return REASONING_KEY.matcher(data).replaceAll("\"reasoning\":");
    }

    private static final Map<String, String> SCHEMA_TYPE_MAP = Map.of(
            "JsonStringSchema",  "string",
            "JsonIntegerSchema", "integer",
            "JsonNumberSchema",  "number",
            "JsonBooleanSchema", "boolean",
            "JsonArraySchema",   "array"
    );

    private Map<String, Object> convertSchemaElements(Map<String, ? extends Object> elements) {
        Map<String, Object> result = new HashMap<>();
        for (Map.Entry<String, ? extends Object> entry : elements.entrySet()) {
            result.put(entry.getKey(), convertElement(entry.getValue()));
        }
        return result;
    }

    private Map<String, Object> convertElement(Object elem) {
        if (elem instanceof JsonObjectSchema schema) {
            Map<String, Object> obj = new HashMap<>();
            obj.put("type", "object");
            obj.put("properties", convertSchemaElements(schema.properties()));
            obj.put("required", schema.required());
            if (schema.description() != null && !schema.description().isEmpty()) {
                obj.put("description", schema.description());
            }
            return obj;
        }

        // All other schema types (string, integer, number, boolean, array...)
        Map<String, Object> typeMap = new HashMap<>();
        typeMap.put("type", SCHEMA_TYPE_MAP.getOrDefault(elem.getClass().getSimpleName(), "string"));
        if (elem instanceof JsonSchemaElement jsonElem
                && jsonElem.description() != null
                && !jsonElem.description().isEmpty()) {
            typeMap.put("description", jsonElem.description());
        }
        return typeMap;
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
                    content = ai.text() != null ? ai.text() : "";
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

        // Build tools array from ToolSpecifications
        List<Map<String, Object>> tools = toolSpecs.stream()
            .map(spec -> {
                Map<String, Object> tool = new HashMap<>();
                tool.put("type", "function");
                Map<String, Object> function = new HashMap<>();
                function.put("name", spec.name());
                function.put("description", spec.description());
                // Build parameters from JsonObjectSchema
                JsonObjectSchema schema = spec.parameters();
                if (schema != null) {
                    Map<String, Object> params = new HashMap<>();
                    params.put("type", "object");
                    params.put("properties", convertSchemaElements(schema.properties()));
                    params.put("required", schema.required());
                    function.put("parameters", params);
                }
                tool.put("function", function);
                return tool;
            })
            .toList();

        System.out.println("tools = " + tools);

        Map<String, Object> body = new HashMap<>();
        body.put("model", "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit");
        body.put("messages", apiMessages);
        body.put("tools", tools);
        body.put("tool_choice", "auto");
        body.put("stream", true);
        body.put("temperature", 0.8);
        body.put("max_tokens", 8192);
        body.put("min_p", 0.06);
        body.put("presence_penalty", 1.2);
        body.put("repeat_penalty", 1.05);
        body.put("chat_template_kwargs", Map.of("enable_thinking", true));
        return body;
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
        return new ArrayList<>(messages.stream()
            .map(m -> switch (m.role()) {
                case "system"    -> (ChatMessage) SystemMessage.from(m.content() + TOOL_INSTRUCTION);
                case "assistant" -> AiMessage.from((String) m.content());
                default          -> UserMessage.from((String) m.content());
            })
            .toList());
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

    /**
     * Accumulates tool call data from SSE chunks.
     * Handles two formats:
     * 1. Streaming delta format: delta.tool_calls[{index, id, function: {name, arguments}}]
     * 2. Message format (mlx-openai-server): message.tool_calls[{id, type, function: {name, arguments}}]
     */
    private static class ToolCallAccumulator {
        private final Map<String, ToolCallBuilder> builders = new HashMap<>();
        private final List<ToolExecutionRequest> toolCalls = new ArrayList<>();
        private boolean complete = false;

        private static class ToolCallBuilder {
            String id;
            String name;
            StringBuilder arguments = new StringBuilder();

            void addFunctionCall(String name, String args) {
                if (name != null) {
                    this.name = this.name != null ? this.name + name : name;
                }
                if (args != null) {
                    this.arguments.append(args);
                }
            }
        }

        // Handle streaming delta format: delta.tool_calls with index field
        public void processDeltaToolCalls(JsonNode toolCallsNode) {
            for (JsonNode tc : toolCallsNode) {
                JsonNode indexNode = tc.path("index");
                String key = indexNode.isInt() ? String.valueOf(indexNode.asInt()) : "0";

                ToolCallBuilder builder = builders.computeIfAbsent(key, k -> new ToolCallBuilder());

                JsonNode idNode = tc.path("id");
                if (idNode.isTextual()) {
                    builder.id = idNode.asText();
                }

                JsonNode functionNode = tc.path("function");
                JsonNode nameNode = functionNode.path("name");
                JsonNode argsNode = functionNode.path("arguments");

                String name = nameNode.isTextual() ? nameNode.asText() : null;
                String args = argsNode.isTextual() ? argsNode.asText() : null;

                builder.addFunctionCall(name, args);
            }
        }

        // Handle message format: message.tool_calls with id, type, function structure
        public void processMessageToolCalls(JsonNode toolCallsNode) {
            for (JsonNode tc : toolCallsNode) {
                String key = tc.path("id").isTextual() ? tc.path("id").asText() : "0";

                ToolCallBuilder builder = builders.computeIfAbsent(key, k -> new ToolCallBuilder());

                JsonNode idNode = tc.path("id");
                if (idNode.isTextual()) {
                    builder.id = idNode.asText();
                }

                JsonNode functionNode = tc.path("function");
                JsonNode nameNode = functionNode.path("name");
                JsonNode argsNode = functionNode.path("arguments");

                String name = nameNode.isTextual() ? nameNode.asText() : null;
                String args = argsNode.isTextual() ? argsNode.asText() : null;

                builder.addFunctionCall(name, args);
            }
        }

        public void markComplete() {
            complete = true;
            for (Map.Entry<String, ToolCallBuilder> entry : builders.entrySet()) {
                ToolCallBuilder builder = entry.getValue();
                ToolExecutionRequest req = ToolExecutionRequest.builder()
                    .id(builder.id)
                    .name(builder.name)
                    .arguments(builder.arguments.toString())
                    .build();
                toolCalls.add(req);
            }
        }

        public List<ToolExecutionRequest> getToolCalls() {
            return toolCalls;
        }
    }
}
