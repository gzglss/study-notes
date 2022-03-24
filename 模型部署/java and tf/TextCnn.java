package com.xiaomi.ai.skill.todofeatureSkill.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;


public class TextCnn {
    private static final Logger LOGGER=LoggerFactory.getLogger(TextCnn.class);
    private static SavedModelBundle bundle=null;
    private static Object tokenid=null;
    final static String modelDir="michatclassifier/pbmodel/";
    final static String vocabDir="michatclassifier/vocab.txt";
    private static Session session=null;


    public static SavedModelBundle loadModel(String modelPath){
        LOGGER.info("loadg model ...:"+modelPath);
        SavedModelBundle savedModelBundle=SavedModelBundle.load(modelPath,"serve");
        return savedModelBundle;
    }

    public static Object word2Id(File file){
        System.out.println(file);
        Map<String,Integer> map=new HashMap<>();
        try{
           BufferedReader br=new BufferedReader(new FileReader(file));
           String s=null;
           int idx=0;
           while ((s=br.readLine())!=null){
               if(!(map.containsKey(s))){
                   map.put(s,idx);
               }
               idx++;
           }
        }catch (Exception e){
            e.printStackTrace();
        }
        return map;
    }

    public void init(){
        String path = this.getClass().getResource("/").getPath();
        System.out.println(path);
        LOGGER.info(path);

        bundle=loadModel(path+modelDir);
        session=bundle.session();
        LOGGER.info("success load");
        tokenid=word2Id(new File(String.valueOf(path+vocabDir)));
    }

//    public void testInit(String path){
//        bundle=loadModel(path+modelDir);
//        session=bundle.session();
//        LOGGER.info("success load");
//        tokenid=word2Id(vocabDir);
//        LOGGER.info("word to id success");
//    }

    public static Object getWordid(String query,Map map,Integer maxLen){
        int[][] arrayList=new int[1][32];
        for(int i=0;i<1;i++){
            for(int j=0;j<32;j++){
                arrayList[i][j]=0;
            }
        }
        for(int i=0;i<maxLen;i++){
            if(i>=query.length()){
                break;
            }
            if(map.containsKey(query.substring(i,i+1))) {
                arrayList[0][i]= (int) map.get(query.substring(i, i + 1));
            }
        }
        return arrayList;
    }

    public boolean getScore(String query) {
        Map<String, Float> scoreMap = new HashMap<>();
//        ArrayList word2id = (ArrayList) getWordid(query, (Map) tokenid, 32);
        Object wordid = getWordid(query, (Map) tokenid, 32);
        float[] keepProb = {1.0f};
        Tensor queryTensor = Tensor.create(wordid);
//        Tensor lengthTensor=Tensor.create(32);
        Tensor keepProbTensor = Tensor.create(keepProb);
        List<Tensor<?>> results = session.runner()
                .feed("input_x", queryTensor)
                .feed("dropout_keep_out", keepProbTensor)
                .fetch("output/scores")
                .fetch("output/predicts")
                .run();
        float[][] score = (float[][]) results.get(0).copyTo(new float[1][2]);
        long[] pred = (long[]) results.get(1).copyTo(new long[1]);

        closeAllTensor(results);
        queryTensor.close();
        keepProbTensor.close();

        float a=0.0f;
        float b=0.0f;
        ArrayList<Float> scoreList=new ArrayList<>();
        for (int i=0; i < score[0].length; i++) {
            scoreList.add((float) score[0][i]);
            System.out.println(scoreList);
        }
        a=scoreList.get(1);
        b=scoreList.get(0);
        return a>b;
    }


    public static void closeAllTensor(final List<Tensor<?>> fetchs){
        for (final Tensor<?> t:fetchs){
            try{
                t.close();
            }catch (final Exception e){
                LOGGER.info("error in closing Tensor {}",e);
            }
        }
        fetchs.clear();
    }
}
