package com.xiaomi.ai.skill.todofeatureSkill.model;

public class tempTextCnn {
    public static void main(String[] args) {
        TextCnn model=new TextCnn();
        model.init();
        Object r = model.getScore("你吃葡萄干吗");

        System.out.println(r);
    }
}
