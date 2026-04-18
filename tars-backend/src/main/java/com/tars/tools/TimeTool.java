package com.tars.tools;

import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

@Component
public class TimeTool {

    private static final ZoneId ZONE = ZoneId.of("Europe/Madrid");
    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("EEEE, d MMMM yyyy 'at' HH:mm");

    @Tool("Returns the current date and time in the user's local timezone (Europe/Madrid, Spain)")
    public String currentTime() {
        return LocalDateTime.now(ZONE).format(FMT);
    }
}
