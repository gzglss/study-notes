package com.xiaomi.ai.skill.todofeatureSkill.model;

import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;


public class lrClassifier{
    //导入模型
    public static Evaluator loadPmml(String path){
        PMML pmml=new PMML();
        InputStream inputStream=null;
        File file=new File(path);
        //建立文件实例
        try{
            inputStream=new FileInputStream(file);
        }catch (Exception e){
            e.printStackTrace();
        }
        if(inputStream==null){
            return null;
        }
        //读取文件
        try{
            pmml=org.jpmml.model.PMMLUtil.unmarshal(inputStream);
        }catch (JAXBException | SAXException e1){
            e1.printStackTrace();
        }
        //关闭文件
        try{
            inputStream.close();
        }catch (IOException e){
            e.printStackTrace();
        }
        ModelEvaluatorFactory modelEvaluatorFactory=ModelEvaluatorFactory.newInstance();
        Evaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        pmml=null;
        return evaluator;
    }

    //模型预测
    public static Object classifier(Evaluator model,Map<String,String> map){
        List<InputField> inputFields=model.getInputFields();//获取模型输入域
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();//存储模型参数
        //导入模型参数
        for (InputField inputField:inputFields){
            FieldName inputFieldName=inputField.getName();//参数名
            FieldValue inputValue=inputField.prepare(map.get(inputFieldName.getValue()));//句子输入
            arguments.put(inputFieldName,inputValue);
        }
        System.out.println(arguments);
        //根据参数预测
        Map<FieldName,?> result=model.evaluate(arguments);
        System.out.println(result);
        List<TargetField> targetFields = model.getTargetFields();

        Map<String, Object> resultMap = new HashMap<>();
        for(TargetField targetField : targetFields) {
            FieldName targetFieldName = targetField.getName();
//            System.out.println("targetFieldName"+targetFieldName);
            Object targetFieldValue = result.get(targetFieldName);
            if (targetFieldValue instanceof Computable) {
                Computable computable = (Computable) targetFieldValue;
                resultMap.put(targetFieldName.getValue(), computable.getResult());
            }else {
                resultMap.put(targetFieldName.getValue(), targetFieldValue);
            }
        }
        return resultMap;
    }
}