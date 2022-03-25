package com.xiaomi.ai.skill.todofeatureSkill.model;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by qijianwei (qijianwei@xiaomi.com)
 * On 22-3-25
 */
public class chatClassifierTest {

    @Test
    public void classifyTest() {
        TextCnn model = new TextCnn();
        model.init();

        Map<String, Boolean> evalSet = new HashMap<>();
        evalSet.put("你不是说看不见吗", true);
        evalSet.put("平和会不会下雨", false);
        evalSet.put("十一点二十提醒我炒菜", false);
        evalSet.put("怎么可能不困", true);
        evalSet.put("来一首我的追求", false);
        evalSet.put("你是不是小乌龟", true);

        evalSet.forEach((query, label) -> {
           Boolean predictLabel = model.isChat(query, "mock");
           assert predictLabel.equals(label);
        });
    }
}
