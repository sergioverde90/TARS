package com.tars.api;

import com.tars.api.dto.ChatCompletionRequest;
import com.tars.chat.TarsChatService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final TarsChatService chatService;

    public ChatController(TarsChatService chatService) {
        this.chatService = chatService;
    }

    @PostMapping(value = "/v1/chat/completions", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> chat(@RequestBody ChatCompletionRequest req) {
        log.info("Chat request: {} messages, stream={}", req.messages().size(), req.isStream());
        return chatService.stream(req);
    }

    @GetMapping("/health")
    public String health() {
        return "ok";
    }
}
