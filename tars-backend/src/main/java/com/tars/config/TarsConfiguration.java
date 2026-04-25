package com.tars.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.tools.DistanceTool;
import com.tars.tools.SearchTool;
import com.tars.tools.TimeTool;
import com.tars.tools.WebsiteScrappingTool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Configuration
public class TarsConfiguration {

    @Bean
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }

    @Bean
    public List<ToolSpecification> toolSpecifications(
            TimeTool timeTool,
            DistanceTool distanceTool,
            SearchTool searchTool,
            WebsiteScrappingTool websiteScrappingTool) {

        ArrayList<ToolSpecification> toolSpecs = new ArrayList<>();
        toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(timeTool));
        toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(distanceTool));
        toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(searchTool));
        toolSpecs.addAll(ToolSpecifications.toolSpecificationsFrom(websiteScrappingTool));

        //log.info("Registered tool specs: {}", toolSpecs.stream().map(ToolSpecification::name).toList());

        return toolSpecs;
    }

    @Bean
    public Map<String, Function<String, String>> toolHandlers(
            ObjectMapper mapper, TimeTool timeTool,
            DistanceTool distanceTool,
            SearchTool searchTool,
            WebsiteScrappingTool websiteScrappingTool) {
        Map<String, Function<String, String>> toolHandlers = new HashMap<>();
        toolHandlers.put("currentTime", args -> timeTool.currentTime());
        toolHandlers.put("scrape", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return websiteScrappingTool.scrape(node.path("url").asText());
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        });
        toolHandlers.put("drivingDistance", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return distanceTool.drivingDistance(
                        node.path("from").asText(),
                        node.path("to").asText()
                );
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });
        toolHandlers.put("webSearch", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return searchTool.webSearch(node.path("query").asText());
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });

        return toolHandlers;
    }

}
