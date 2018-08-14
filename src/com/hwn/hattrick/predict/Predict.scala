package com.hwn.hattrick.predict

import java.util.Date

import com.redis._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import java.net.URLDecoder

object Predict {

  def main(args: Array[String]) {
    val trainingDataOrModelPath = args(0)
    val redisHost = args(1)
    val redisChannelReceive = args(2)
    val redisChannelResponse = args(3)

    val spark = SparkSession.builder
      .master("local")
      .appName("hattrick")
      //      .config("spark.some.config.option", "config-value")
      .getOrCreate()
    val model = if (redisHost.equals("127.0.0.1")) getModel(trainingDataOrModelPath) else createModel(trainingDataOrModelPath)

    val redisReceive = new RedisClient(redisHost, 6379)
    val redisResponse = new RedisClient(redisHost, 6379)

    redisReceive.subscribe(redisChannelReceive) { m =>
      m match {
        case S(channel, no) => println("subscribed to " + channel + " and count = " + no)
        case U(channel, no) => println("unsubscribed from " + channel + " and count = " + no)
        case E(msg)         => println("error msg = " + msg)
        case M(channel, msg) => {
          try {
            println(msg)
            val playerId = msg.substring(0, msg.indexOf(" "))
            val data = msg.substring(msg.indexOf(" ") + 1)
            val time1 = new Date().getTime
            val price = TrainRandomForest.predict(spark, model, playerId, data)
            val time2 = new Date().getTime
            println("time prediction: " + (time2 - time1))
            redisResponse.publish(redisChannelResponse, playerId + " " + price)
          } catch {
            case e: Exception => println("Exception = " + e.getMessage)
          }
        }
      }
    }

    Thread.sleep(999999999)
    println("EXIT")
  }

  def getModel(modelPath: String): PipelineModel = {
    val model = PipelineModel.load(modelPath)
    model
  }

  def createModel(trainingDataPath: String): PipelineModel = {
    TrainRandomForest.getModel(Array(trainingDataPath))
  }

}