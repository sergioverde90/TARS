package com.tars;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class TarsApplication {
    public static void main(String[] args) {
        SpringApplication.run(TarsApplication.class, args);
    }
}
