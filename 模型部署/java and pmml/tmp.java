package com.xiaomi.ai.skill.todofeatureSkill.model;

//import com.huaban.analysis.jieba.JiebaSegmenter;
//import com.huaban.analysis.jieba.WordDictionary;
import org.jpmml.evaluator.Evaluator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


import static com.xiaomi.ai.skill.todofeatureSkill.model.lrClassifier.loadPmml;
import static com.xiaomi.ai.skill.todofeatureSkill.model.lrClassifier.classifier;

public class tmp {
    public static void main(String[] args) {
        Map<String,String> map=new HashMap<>();//加入数据
//        JiebaSegmenter segment = new JiebaSegmenter();
        ArrayList<String> arrayList=new ArrayList<>();//存储每个word
        String sentence="\\xe9\\xa2\\x9c\\xe8\\x89\\xb2 \\xe4\\xb8\\x80\\xe7\\x82\\xb9 \\xe4\\xb8\\x8a\\xe7\\x8f\\xad";
//        for(int i=0;i<sentence.length();i++) {
//            arrayList.add(String.valueOf(sentence.charAt(i)));
//        }
//        String inputs = String.join(" ", arrayList);//将句子转化为word1 word2 ...的形式
        map.put("query",sentence);
        System.out.println(map.get("query"));
        Evaluator evaluator=loadPmml("app/com/xiaomi/ai/skill/todofeatureSkill/model/classifier_f1_0.827.pmml");//导入模型
        System.out.println(evaluator);
        assert evaluator != null;
        Object pred=classifier(evaluator,map);//模型预测
        System.out.println(pred);
    }
}
