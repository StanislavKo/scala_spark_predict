package com.hwn.hattrick.predict

import java.io._

import scala.collection.mutable

import org.apache.log4j._

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ IntegerType, DoubleType }
import org.apache.spark.sql.{ Row, Dataset }
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{ Transformer, Pipeline, PipelineStage, PipelineModel }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{ StringIndexer, StringIndexerModel, VectorAssembler }
import org.apache.spark.ml.regression.{ RandomForestRegressionModel, RandomForestRegressor }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

object TrainRandomForest {

  def main(args: Array[String]) {
    getModel(args)
  }

  def getModel(args: Array[String]): PipelineModel = {
    Logger.getLogger("org").setLevel(Level.WARN)
    //    val sc = new SparkContext("local[*]", "Hattrick")
    val inPath = if (args.length == 1) args(0) else "g:/hattrick_sold.csv"

    //    val input = sc.textFile("../spark_files/hattrick/hattrick_sold.csv")
    val spark = SparkSession.builder
      .master("local")
      .appName("hattrick")
      //      .config("spark.some.config.option", "config-value")
      .getOrCreate()

    val data1 = spark.read.option("header", "true").csv(inPath)
    val data2 = data1.withColumn("PriceTmp", expr("Price").cast(IntegerType)).drop("Price").withColumnRenamed("PriceTmp", "Price")
      .withColumn("ExperienceTmp", expr("Experience").cast(IntegerType)).drop("Experience").withColumnRenamed("ExperienceTmp", "Experience")
      .withColumn("LidershipTmp", expr("Lidership").cast(IntegerType)).drop("Lidership").withColumnRenamed("LidershipTmp", "Lidership")
      .withColumn("FormTmp", expr("Form").cast(IntegerType)).drop("Form").withColumnRenamed("FormTmp", "Form")
      .withColumn("AgeTmp", expr("Age").cast(DoubleType)).drop("Age").withColumnRenamed("AgeTmp", "Age")
      .withColumn("TSITmp", expr("TSI").cast(IntegerType)).drop("TSI").withColumnRenamed("TSITmp", "TSI")
      .withColumn("StaminaTmp", expr("Stamina").cast(IntegerType)).drop("Stamina").withColumnRenamed("StaminaTmp", "Stamina")
      .withColumn("GoalkeeperTmp", expr("Goalkeeper").cast(IntegerType)).drop("Goalkeeper").withColumnRenamed("GoalkeeperTmp", "Goalkeeper")
      .withColumn("HalfbackTmp", expr("Halfback").cast(IntegerType)).drop("Halfback").withColumnRenamed("HalfbackTmp", "Halfback")
      .withColumn("PassTmp", expr("Pass").cast(IntegerType)).drop("Pass").withColumnRenamed("PassTmp", "Pass")
      .withColumn("WingTmp", expr("Wing").cast(IntegerType)).drop("Wing").withColumnRenamed("WingTmp", "Wing")
      .withColumn("DefenseTmp", expr("Defense").cast(IntegerType)).drop("Defense").withColumnRenamed("DefenseTmp", "Defense")
      .withColumn("AttackTmp", expr("Attack").cast(IntegerType)).drop("Attack").withColumnRenamed("AttackTmp", "Attack")
      .withColumn("StandardTmp", expr("Standard").cast(IntegerType)).drop("Standard").withColumnRenamed("StandardTmp", "Standard")
    //    println(data.getClass)
    //    data.head(5).foreach(println)
    //    println(data.head().getAs("Lidership").getClass())
    //    data.dtypes.foreach(println)
    //    data2.dtypes.foreach(println)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data2.randomSplit(Array(0.75, 0.25))

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    //    val featureIndexer1 = new VectorIndexer()
    //      .setInputCol("isSold")
    //      .setOutputCol("indexedFeatures")
    //      .setMaxCategories(6)
    //      .fit(data)
    // Feature types
    val vars_categ = Array("isSold", "Spec", "Ill")
    val vars_num = Array("Experience", "Lidership", "Form", "TSI", "Stamina", "Goalkeeper", "Halfback", "Pass", "Wing", "Defense", "Attack", "Standard")
    val vars_num_double = Array("Age")

    val stringIndexers = vars_categ.map(colName => new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed").fit(data2))

    val assembler = new VectorAssembler()
      .setInputCols(Array("isSold_indexed", "Spec_indexed", "Ill_indexed") ++ vars_num ++ vars_num_double)
      .setOutputCol("features")

//    for (
//      numTrees <- Seq(15, 30);
//      maxDepth <- Seq(20, 30);
//      maxBins <- Seq(20)
//    ) {
//      train(stringIndexers, assembler, trainingData, testData, numTrees, maxDepth, maxBins)
//    }
//    null
        deleteRecursively(new File("g:/temp/12/hattrick/model"))
        deleteRecursively(new File("/home/stanislav/hattrick/model"))
            val model = train(stringIndexers, assembler, trainingData, testData, 15, 20, 20)
//        val model = train(stringIndexers, assembler, trainingData, testData, 2, 15, 20)
        try {
          model.save("file:///home/stanislav/hattrick/model");
        } catch {
          case ioe: IOException =>
          case e: Exception     =>
        }
        model
  }

  def train(stringIndexers: Array[StringIndexerModel], assembler: VectorAssembler, trainingData: Dataset[Row], testData: Dataset[Row], numTrees: Integer, maxDepth: Integer, maxBins: Integer): PipelineModel = {
    print("numTrees = " + numTrees + ", maxDepth = " + maxDepth + ", maxBins = " + maxBins)
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setMaxBins(maxBins)
      .setMaxDepth(maxDepth)
      .setNumTrees(numTrees)
      .setLabelCol("Price")
      .setFeaturesCol("features")

    //    var stages = new mutable.ArrayBuffer[PipelineStage]()
    //    stages = stages ++ stringIndexers

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(stringIndexers ++ Array(assembler) ++ Array(rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    //    predictions.select("prediction", "Price", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("Price")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    //    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    println("    RMSE = " + rmse)

//    val rfModel = model.stages(model.stages.length - 1).asInstanceOf[RandomForestRegressionModel]
//    println("Learned regression forest model:\n" + rfModel.toDebugString)
    model
  }

  def predict(spark: SparkSession, model: Transformer, playerId: String, dataStr1: String): Double = {
    //    val data = collection.mutable.Map[String, String]()
    //    var dataStr = dataStr1
    //    while (dataStr.indexOf("\"") != -1) {
    //      var start = dataStr.indexOf("\"") + 1
    //      var end = dataStr.indexOf("\"", dataStr.indexOf("\"") + 1)
    //      val key = dataStr.substring(start, end)
    //      dataStr = dataStr.substring(end + 1)
    //      start = dataStr.indexOf("\"") + 1
    //      end = dataStr.indexOf("\"", dataStr.indexOf("\"") + 1)
    //      val value = dataStr.substring(start, end)
    //      dataStr = dataStr.substring(end + 1)
    //      data.put(key, value)
    //    }
    ////    for ((k,v) <- data) printf("key: %s, value: %s\n", k, v)
    //    return data("Attack").toDouble
    val rdd = spark.sparkContext.parallelize(Seq(dataStr1))
    val data = spark.read.json(rdd)
    val row = model.transform(data).head()
    println(row.toString())
    return row.getAs[Double]("prediction")
  }

  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
      file.listFiles.foreach(deleteRecursively)
    if (file.exists && !file.delete)
      throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
  }

}