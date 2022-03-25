package com.xiaomi.ai.skill.todofeatureSkill.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;


public class TextCnn {
    private static final Logger LOGGER=LoggerFactory.getLogger(TextCnn.class);
    private static Map<String, Integer> token2idx;
    private static Session session;

    private static SavedModelBundle loadModel(String modelPath){
        LOGGER.info("Loading model from:"+ modelPath);
        return SavedModelBundle.load(modelPath,"serve");
    }

    private static Map<String, Integer> loadToken2idx(InputStream inputStream){
        Map<String,Integer> map=new HashMap<>();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));

        try{
           String s;
           int idx = 0;
           while ((s = bufferedReader.readLine())!=null){
               if(!(map.containsKey(s))){
                   map.put(s,idx);
               }
               idx++;
           }
        }catch (Exception e){
            LOGGER.error("Load chat classifier vocab failed.");
            e.printStackTrace();
        }
        return map;
    }

    public void init(){
        SavedModelBundle bundle = loadModel(this.getClass().getResource("/model/chatClassifier").getPath());
        session = bundle.session();
        LOGGER.info("success load");
        token2idx = loadToken2idx(this.getClass().getResourceAsStream("/model/chatClassifier/vocab.txt"));
    }

    private static int[][] getTokenIds(String query){
        int maxLength = 32;

        int[][] tokenIds=new int[1][maxLength];

        for (int i=0; i<maxLength; i++){
            if (i < query.length()) {
                tokenIds[0][i] = token2idx.getOrDefault(query.substring(i, i + 1), token2idx.get("[UNK]"));
            }
        }
        return tokenIds;
    }

    public Boolean isChat(String query, String requestId) {
        int[][] tokenIds = getTokenIds(query);
        float[] keepProb = {1.0f};
        Tensor queryTensor = Tensor.create(tokenIds);
        Tensor keepProbTensor = Tensor.create(keepProb);
        List<Tensor<?>> results = session.runner()
                .feed("input_x", queryTensor)
                .feed("dropout_keep_out", keepProbTensor)
                .fetch("output/scores")
                .fetch("output/predicts")
                .run();
        float[][] score = results.get(0).copyTo(new float[1][2]);

        closeAllTensor(results);
        queryTensor.close();
        keepProbTensor.close();
        LOGGER.info("requestId: {}, query: {}, score1: {}, score0: {}", requestId, query, score[0][1], score[0][0]);
        return score[0][1] > score[0][0];
    }


    private static void closeAllTensor(final List<Tensor<?>> fetchs){
        for (final Tensor<?> t:fetchs){
            try{
                t.close();
            }catch (final Exception e){
                LOGGER.error("Error in closing Tensor ", e);
            }
        }
        fetchs.clear();
    }
}
