package com.tars.api;

import com.tars.api.dto.ChatCompletionRequest;
import com.tars.chat.ChatCompletionService;
import com.tars.chat.StreamChatCompletionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final ChatCompletionService chatCompletionService;
    private final StreamChatCompletionService streamChatCompletionService;

    public ChatController(
        ChatCompletionService chatCompletionService,
        StreamChatCompletionService streamChatCompletionService
    ) {
        this.chatCompletionService = chatCompletionService;
        this.streamChatCompletionService = streamChatCompletionService;
    }

    @PostMapping("/v1/chat/completions")
    public Object chat(@RequestBody ChatCompletionRequest req) {
        log.info("Chat request: {} messages, stream={}", req.messages().size(), req.isStream());
        if (req.isStream()) {
            return streamChatCompletionService.stream(req);
        }
        return ResponseEntity.ok(chatCompletionService.chat(req));
    }

    @GetMapping("/health")
    public String health() {
        return "ok";
    }
}
