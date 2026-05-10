package com.tars.tools;

import org.springframework.stereotype.Component;

import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

@Component
public class TimeTool {
    public String currentTime() {
        return DateTimeFormatter.ISO_ZONED_DATE_TIME.format(ZonedDateTime.now());
    }
}
