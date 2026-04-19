package com.tars.api;

import com.tars.api.dto.ChatCompletionRequest;
import com.tars.chat.TarsChatService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final TarsChatService chatService;

    public ChatController(TarsChatService chatService) {
        this.chatService = chatService;
    }

    @PostMapping("/v1/chat/completions")
    public Mono<ResponseEntity<?>> chat(@RequestBody ChatCompletionRequest req) {
        log.info("Chat request: {} messages, stream={}", req.messages().size(), req.isStream());
        if (req.isStream()) {
            return Mono.just(ResponseEntity.ok()
                .contentType(MediaType.TEXT_EVENT_STREAM)
                .body(chatService.stream(req)));
        }
        return chatService.chat(req).map(ResponseEntity::ok);
    }

    @GetMapping("/health")
    public String health() {
        return "ok";
    }
}
