


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.List;
import java.util.Map;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;


public class GenerateLTSM  {
	
	private static int filesNumber;
	private static List<Integer> elementsInFile = new ArrayList<Integer>();
	private static List<Integer> linesInFile = new ArrayList<Integer>();
	
    private static String folderName = "levels/original/";

    private Random rnd;
    private static HashMap<Character, Integer> dict = new HashMap<Character, Integer>();
    private static List<List<List<Integer>>> inputArray;
    
    
    
    
    private static void prepareData(List<HashMap<Integer, Character>> dataFromLines) {
    	int dl = dict.size();
    	inputArray = new ArrayList<List<List<Integer>>>();
    	int k = 0;   	
    	for (HashMap<Integer, Character> data : dataFromLines) {
    		List<List<Integer>> tempArray = new ArrayList<List<Integer>>();
    		for (int i =0; i<elementsInFile.get(k); i++) {tempArray.add(new ArrayList<Integer>());}
    		
    		for (Map.Entry<Integer, Character> entry : data.entrySet()) {
    			List<Integer> tempArrayLoc = new ArrayList<Integer>();
    			for (int i = 0; i< dl; i++) {tempArrayLoc.add(0);}
    			tempArrayLoc.set(dict.get(entry.getValue()), 1);
    			tempArray.set(entry.getKey(), tempArrayLoc);
    			
    		}
    		
    		k++;
    		inputArray.add(tempArray);	
    	}
    	
    }
    private static List<List<Integer>> makeSnake(List<List<Integer>> data, int dir, int file){
    	int step = linesInFile.get(file);
    	//int step = 16;
    	List<Integer> temp; 
    	
    	for (int i = dir*step; i < elementsInFile.get(file); i= i + 2*step) {
    		int k = 0;
    		for (int j =i; j <i + 0.5*step; j++) {
    			temp = data.get(j);
    			
    			data.set(j, data.get(i+step - k - 1));
    			data.set(i+step - k - 1, temp);
    			k++;
    		}
    	}
    	
    	return data;
    }
    private static List<List<Integer>> dataForNet(List<List<Integer>> preparedData, int getprevsteps, int file) {
    	int nExamples = elementsInFile.get(file) - getprevsteps;
    	int nFeatures = dict.size();
    	List<List<Integer>> result= new ArrayList<List<Integer>>();
    	for (int i =0;i< nExamples;i++) {
    		List<Integer> temp = new ArrayList<Integer>();
    		for (int j = 0; j < getprevsteps; j++) {
    			for (int k =0; k< nFeatures; k++) {temp.add(preparedData.get(i+j).get(k));}
    		
    		}
    		
    		for (int k =0; k< nFeatures; k++) {temp.add(preparedData.get(i+getprevsteps).get(k));}
    		result.add(temp);
   	    		
    	}
    	return result;
    }
    private static void writeCSV(List<List<Integer>> data, Integer n) throws IOException {
    	String namefeatures = "levels/my/fetures_"+n.toString()+".csv";
    	String nametargets = "levels/my/targets_"+n.toString()+".csv";
    	FileWriter writer_f = new FileWriter(namefeatures);
    	FileWriter writer_t = new FileWriter(nametargets);
    	for (int i =0;i<data.size();i++) {
    		String s = "";
    		for (int j =0;j<data.get(i).size()-dict.size()-1;j++) {
    			s=s+data.get(i).get(j).toString();
    			s=s+",";
    		}
    		s = s+data.get(i).get(data.get(i).size()-dict.size()-1).toString();
    		if(i<data.size()-1) s = s+"\n";
    		writer_f.write(s);
    		
    		s = "";
    		for (int j =data.get(i).size()-dict.size();j<data.get(i).size()-1;j++) {
    			s=s+data.get(i).get(j).toString();
    			s=s+",";
    		}
    		s = s+data.get(i).get(data.get(i).size()-1).toString();
    		if(i<data.size()-1) s = s+"\n";
    		writer_t.write(s);
    		
    	}
    
    	 writer_f.close();
    	 writer_t.close();
    }
    private static void savePictureToFile(String filename, List<List<Integer>> data) throws FileNotFoundException, UnsupportedEncodingException {
       List<String> lines_new = makePictureFile(makeSnake(data,1,1),linesInFile.get(1));
 	   PrintWriter writer = new PrintWriter(filename, "UTF-8");
 	   for (int k = 0;k<linesInFile.get(1)-1; k++) {
 		   writer.println(lines_new.get(k));
 	   }
 	   writer.print(lines_new.get(linesInFile.get(1)-1));
 	   writer.close();   
    }
    
    private static void prepareAndwrite() throws IOException {
    	File[] listOfFiles = new File(folderName).listFiles();
    	List<HashMap<Integer, Character>> data = new ArrayList<HashMap<Integer, Character>>();
    	List<String> lines;
    	Set<Character> unique =new HashSet<Character>();
    	filesNumber = listOfFiles.length;     		
        for (int i = 0; i < filesNumber; i++ ) {
        	
        	lines = Files.readAllLines(listOfFiles[i].toPath());
        	int nls = lines.size();
        	
        	HashMap<Integer,Character> hmap =new HashMap<Integer,Character>();
        	int j = 0;
        	int temp=0;
        	for (String line : lines) {
        		int k = 0;
        		int nl = line.length();
        		temp+=nl;
        		for (Character ch : line.toCharArray()) {
        		hmap.put(nls - j - 1 + k*nls, ch);
        		unique.add(ch);
        		k+=1;
        		}
        		
        		j+=1;
        	}
        	
        	elementsInFile.add(temp);
        	linesInFile.add(nls);
        	data.add(hmap);
        	
        	
        }
        
             
       int j = 0;
       for (Character ch : unique) {
    	   dict.put(ch,j);
    	   j++;
       }
       
      prepareData(data);
     
       for (int i =0;i<filesNumber;i++) {
    	  
      
    	   List<List<Integer>> s0 = makeSnake(new ArrayList(inputArray.get(i)), 0, i);
    	   List<List<Integer>> d0 = dataForNet(s0,1,i);
    	   writeCSV(d0,2*i);
    	   List<List<Integer>> s1 = makeSnake(new ArrayList(inputArray.get(i)), 1, i);
    	   List<List<Integer>> d1 = dataForNet(s1,1,i);
    	   writeCSV(d1,2*i+1);
       }
      
    }
    
    private static List<String> makePictureFile(List<List<Integer>> data, Integer rowN) {
    	HashMap<Integer,Character > tempdict = new HashMap<Integer,Character>();
    	for (Map.Entry<Character,Integer> entry : dict.entrySet()) {tempdict.put(entry.getValue(), entry.getKey()); }
    	
    	List<String> lines = new ArrayList<String>();
    	for (int i=0;i<rowN;i++) {
    		lines.add(new String());
    		lines.set(i,"");
    	}
    	
    	for (int i = 0; i< data.size();i++) {
    		int j =i%rowN;
    		
    		lines.set(rowN-1-j, lines.get(rowN-1-j)+tempdict.get(data.get(i).indexOf(1)).toString());
    		
    	}
    	return lines;
    	
    }
    private static INDArray predict(INDArray input, MultiLayerNetwork net){
    	
    	
    	INDArray timeSeriesOutput = net.rnnTimeStep(input);
    	//System.out.println(timeSeriesOutput);
	    INDArray lastTimeStepIndices = Nd4j.argMax(timeSeriesOutput,1);
	    INDArray result =Nd4j.zeros(1,timeSeriesOutput.size(1));
		result.putScalar(0,lastTimeStepIndices.getInt(), 1);
	    return result; 
	    }

    	
     
  
    private static SequenceRecordReaderDataSetIterator getFromCSV() throws IOException{
    	CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, ",");
    	try {
			trainFeatures.initialize( new NumberedFileInputSplit("levels/my" + "/fetures_%d.csv", 0, 29));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	CSVSequenceRecordReader trainTargets = new CSVSequenceRecordReader(0, ",");
    	try {
			trainTargets.initialize(new NumberedFileInputSplit("levels/my" + "/targets_%d.csv", 0,29));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

    	SequenceRecordReaderDataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainTargets, 100,
    	                30, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);
    	                
    return train;
    }
    
    private static MultiLayerNetwork neuralNet() {
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    		    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    		    .seed(12345)
    		    .dropOut(0.25)
    		    .weightInit(WeightInit.XAVIER)
    		    .updater(new Adam())
    		    .list()
    		    .layer(0, new LSTM.Builder()
    		        .activation(Activation.TANH)
    		        //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    		        //.gradientNormalizationThreshold(10)
    		        .nIn(30)
    		        .nOut(512)
    		        .build())
    		    .layer(1, new LSTM.Builder()
        		        .activation(Activation.TANH)
        		        //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        		        //.gradientNormalizationThreshold(10)
        		        .nIn(512)
        		        .nOut(512)
        		        .build())
    		    .layer(2, new LSTM.Builder()
        		        .activation(Activation.TANH)
        		        //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        		        //.gradientNormalizationThreshold(10)
        		        .nIn(512)
        		        .nOut(512)
        		        .build())
    		    
    		    .layer(3, new RnnOutputLayer.Builder(LossFunction.MCXENT)
    		        .activation(Activation.SOFTMAX)
    		        .nIn(512)
    		        .nOut(30)
    		        .build())
    		.backpropType(BackpropType.TruncatedBPTT)
    		.tBPTTLength(200)
    		.build();
    		    
    	 MultiLayerNetwork net = new MultiLayerNetwork(conf);
    	 return net;
    		
    	
    }
  
    

	public static void main(String[] args) throws IOException, InterruptedException {
		prepareAndwrite();
    	SequenceRecordReaderDataSetIterator train = getFromCSV();
    	
    	MultiLayerNetwork net = neuralNet();
    	
    	net.init();
    	
    	net.fit(train,200);
    	System.out.println("done");
    	CSVSequenceRecordReader tFeatures = new CSVSequenceRecordReader(3215, ",");
    	
			tFeatures.initialize( new NumberedFileInputSplit("levels/my" + "/fetures_%d.csv", 0, 0));
		
			CSVSequenceRecordReader tTargets = new CSVSequenceRecordReader(3215, ",");
	    	tTargets.initialize(new NumberedFileInputSplit("levels/my" + "/targets_%d.csv", 0,0));
	    	System.out.println(tTargets.toString());
			
	    	SequenceRecordReaderDataSetIterator t = new SequenceRecordReaderDataSetIterator(tFeatures, tTargets, 1,
	    	                30, true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);
	    	
	    	
	    	
	    	net.rnnClearPreviousState();
	    	List<List<Integer>> picture = makeSnake(new ArrayList(inputArray.get(1)), 1, 1);
	    	
	    	INDArray test =Nd4j.zeros(1,30);
    		for (int i=0;i<32;i++) {
    			test.putScalar(0,picture.get(i).indexOf(1),1);
    			INDArray ss=predict(test, net);
    			//System.out.println(ss);
    			test = ss;
    		}
	    	
    		
    		for (int i=0;i<elementsInFile.get(1)-32;i++) {
    			List<Integer> temp = new ArrayList<Integer>();
    			for (int j=0;j<30;j++) {temp.add(0);}
    			INDArray ss=predict(test, net);
    			temp.set(Nd4j.argMax(ss,1).getInt() ,1);
    			
    			test = ss;
    			picture.set(i+32,temp);
    		
    		}
    		
    		savePictureToFile("levels/my/filename.txt", picture);
	}
	
	}

