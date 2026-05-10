package com.tars.api;

import com.tars.api.dto.ChatCompletionRequest;
import com.tars.chat.StreamChatCompletionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.*;

@CrossOrigin
@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final StreamChatCompletionService streamChatCompletionService;

    public ChatController(
        StreamChatCompletionService streamChatCompletionService
    ) {
        this.streamChatCompletionService = streamChatCompletionService;
    }

    @PostMapping("/v1/chat/completions")
    public Object chat(@RequestBody ChatCompletionRequest req) {
        log.info("Chat request: {} messages, stream={}", req.messages().size(), req.isStream());
        return streamChatCompletionService.stream(req);
    }

    @GetMapping("/health")
    public String health() {
        return "ok";
    }
}
