package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class describes a fitting run.  It is used to communicate between the training-related
 * processors and the actual trainers.
 *
 * @author Bruce Parrello
 */
abstract public class RunStats {

    /** Method for determining classification model to save */
    public static enum OptimizationType {
        /** save the model with the lowest score */
        SCORE,
        /** save the model with the highest accuracy */
        ACCURACY;
    }

    /** Method for determining regression model to save */
    public static enum RegressionType {
        /** save the model with the lowest score */
        SCORE,
        /** save the model with the highest coefficient of determination */
        RSQUARED,
        /** save the model with the highest Pearson correlation */
        PEARSON,
        /** save the model with the highest pseudo-accuracy */
        ACCURACY
        ;
    }

    /** logging facility */
    public static Logger log = LoggerFactory.getLogger(ClassTrainingProcessor.class);

    // FIELDS

    /** TRUE if we stopped because of an error, else FALSE */
    private boolean errorStop;

    /** number of times the score bounced */
    private int bounceCount;

    /** number of training events */
    private int eventCount;

    /** saved copy of the best model found */
    private MultiLayerNetwork bestModel;

    /** number of times the model was saved */
    private int saveCount;

    /** event count when the model was saved */
    private int bestEvent;

    /** accuracy when the model was saved */
    private double bestAccuracy;

    /** number of events in a row where the model was not saved */
    private int uselessEvents;

    /** score when the model was saved */
    private double bestScore;

    /** name of the relevant training event */
    private String eventsName;

    /** displayable duration of training run */
    private String duration;

    /** most recent accuracy */
    protected double newAccuracy;

    /** most recent score */
    protected double newScore;

    /** output from best model */
    protected INDArray output;

    /**
     * Construct a blank run tracker.
     *
     * @param model		model whose runs are to be tracked
     */
    protected RunStats(MultiLayerNetwork model) {
        this.errorStop = false;
        this.bounceCount = 0;
        this.eventCount = 0;
        this.bestModel = model;
        this.bestEvent = 0;
        this.saveCount = 0;
        this.bestAccuracy = -Double.MAX_VALUE;
        this.uselessEvents = 0;
        this.bestScore = Double.MAX_VALUE;
        this.duration = "00:00";
    }

    /**
     * Construct a classification run tracker of the appropriate type.
     *
     * @param model		model whose runs are to be tracked
     * @param type		desired optimization type
     * @oaram trainer	training running the statistics
     */
    public static RunStats create(MultiLayerNetwork model, OptimizationType type, Trainer trainer) {
        RunStats retVal = null;
        switch (type) {
        case SCORE:
            retVal = new RunStats.Score(model);
            break;
        case ACCURACY:
            retVal = new RunStats.Accuracy(model);
            break;
        }
        retVal.eventsName = trainer.eventsName();
        return retVal;
    }

    /**
     * Construct a regression run tracker of the appropriate type.
     *
     * @param model		model whose runs are to be tracked
     * @param type		desired optimization type
     * @oaram trainer	trainer running the statistics
     * @param processor	regression training processor for the model
     */
    public static RunStats createR(MultiLayerNetwork model, RegressionType type, Trainer trainer,
            RegressionTrainingProcessor processor) {
        RunStats retVal = null;
        switch (type) {
        case SCORE:
            retVal = new RunStats.Regression(model, processor);
            break;
        case RSQUARED:
            retVal = new RunStats.Coefficient(model, processor);
            break;
        case PEARSON:
            retVal = new RunStats.Pearson(model, processor);
            break;
        case ACCURACY:
            retVal = new RunStats.PseudoAccuracy(model, processor);
            break;
        }
        retVal.eventsName = trainer.eventsName();
        return retVal;
    }

    /** Record a score bounce. */
    public void bounce() {
        this.bounceCount++;
    }

    /** Record a batch. */
    public void event() {
        this.eventCount++;
    }

    /** Record an error. */
    public void error() {
        this.errorStop = true;
    }

    /**
     * @return the error-stop flag
     */
    public boolean isErrorStop() {
        return errorStop;
    }

    /**
     * @return the number of score bounces
     */
    public int getBounceCount() {
        return bounceCount;
    }

    /**
     * @return the number of events processed
     */
    public int getEventCount() {
        return eventCount;
    }

    /**
     * @return the score for the last model saved
     */
    public double getBestScore() {
        return bestScore;
    }

    /**
     * @return the best model found
     */
    public MultiLayerNetwork getBestModel() {
        return bestModel;
    }

    /**
     * @return the number of model saves
     */
    public int getSaveCount() {
        return saveCount;
    }

    /**
     * @return the event corresponding to the best model
     */
    public int getBestEvent() {
        return bestEvent;
    }

    /**
     * @return the accuracy of the best model
     */
    public double getBestAccuracy() {
        return bestAccuracy;
    }

    /**
     * Store the duration string.
     *
     * @param newValue	duration to store
     */
    public void setDuration(String newValue) {
        this.duration = newValue;
    }

    /**
     * @return the duration string
     */
    public String getDuration() {
        return duration;
    }

    /**
     * Store the new best model.
     * @param bestModel 	the new best model
     */
    public void setBestModel(MultiLayerNetwork bestModel) {
        this.bestModel = bestModel;
        this.bestAccuracy = this.newAccuracy;
        this.bestScore = this.newScore;
        this.bestEvent = this.eventCount;
        this.saveCount++;
        this.uselessEvents = 0;
    }

    /**
     * @return the number of useless iterations so far
     */
    public int getUselessIterations() {
        return this.uselessEvents;
    }

    /**
     * Increment the count of useless iterations.
     */
    public void uselessIteration() {
        this.uselessEvents++;
    }

    /**
     * @return the rating of the best model (accuracy or negative of score)
     */
    public double getBestRating() {
        return getBestAccuracy();
    }

    /**
     * Check the model to see if we need to save this as the best model so far.
     *
     * @param model			current state of the model
     * @param testingSet	testing set for evaluating the model
     * @param processor		training processor managing the training
     * @param seconds		number of seconds spent processing this section
     * @param eventType		label to use for events
     * @param processType	label to use for processing
     */
    abstract public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor,
            double seconds,String eventType, String processType);

    /**
     * Write a report to the trial log.
     *
     * @param logFile	file to contain the trial log
     * @param label		heading comment, if any
     * @param report	text of the report to write, with internal new-lines
     *
     * @throws IOException
     */
    public static void writeTrialReport(File logFile, String label, String report) throws IOException {
        // Open the trials log in append mode and write the information about this run.
        PrintWriter trialWriter = new PrintWriter(new FileWriter(logFile, true));
        trialWriter.println("******************************************************************");
        if (label != null)
            trialWriter.print(label);
        trialWriter.println(report);
        trialWriter.close();
    }

    /**
     * Evaluate a model against a testing set
     *
     * @param model			model to evaluate
     * @param testingSet	testing set for the evaluation
     * @param labels		list of label names
     *
     * @return an evaluation object containing an assessment of the model's performance
     */
    public Evaluation evaluateModel(MultiLayerNetwork model, DataSet testingSet, List<String> labels) {
        this.output = model.output(testingSet.getFeatures());
        Evaluation retVal = new Evaluation(labels);
        retVal.eval(testingSet.getLabels(), this.output);
        this.newAccuracy = retVal.accuracy();
        this.newScore = model.score();
        return retVal;
    }

    /**
     * Statistically score a model against a testing set.  Subclasses that use this must provide
     * a chooseAltScore method.
     *
     * @param model			model to evaluate
     * @param testingSet	testing set for the evaluation
     *
     * @return an evaluation object containing an assessment of the model's performance
     */
    public RegressionEvaluation scoreModel(MultiLayerNetwork model, DataSet testingSet, List<String> labels) {
        this.output = model.output(testingSet.getFeatures());
        RegressionEvaluation retVal = new RegressionEvaluation(labels);
        retVal.eval(testingSet.getLabels(), this.output);
        this.newAccuracy = this.chooseAltScore(retVal, testingSet);
        this.newScore = model.score();
        return retVal;
    }

    /**
     * Compute the pseudo-accuracy for a given column.  The pseudo-accuracy is the fraction of rows
     * where the output value in the column is on the same side of the bound as the value in the testing
     * set.
     *
     * @param testingSet	testing set containing desired results
     * @param col			label column to process
     * @param bound			threshold for pseudo-accuracy
     *
     * @return the pseudo-accuracy for the given label column
     */
    public double pseudoAccuracy(DataSet testingSet, int col, double bound) {
        INDArray expect = testingSet.getLabels();
        int goodCount = 0;
        for (long r = 0; r < expect.rows(); r++) {
            boolean eDiff = (expect.getDouble(r, col) >= bound);
            boolean oDiff = (output.getDouble(r, col) >= bound);
            if (eDiff == oDiff) goodCount++;
        }
        return (((double) goodCount) / expect.rows());
    }

    /**
     * Compute the pseudo-accuracy for all columns.  The pseudo-accuracy is the fraction of entries
     * where the output value is on the same side of the bound as the value in the testing
     * set.
     *
     * @param testingSet	testing set containing desired results
     * @param bound			threshold for pseudo-accuracy
     *
     * @return the pseudo-accuracy for the given label column
     */
    public double pseudoAccuracy(DataSet testingSet, double bound) {
        INDArray expect = testingSet.getLabels();
        int goodCount = 0;
        int total = 0;
        for (long r = 0; r < expect.rows(); r++) {
            for (long c = 0; c < expect.columns(); c++) {
                boolean eDiff = (expect.getDouble(r, c) >= bound);
                boolean oDiff = (output.getDouble(r, c) >= bound);
                if (eDiff == oDiff) goodCount++;
                total++;
            }
        }
        return (((double) goodCount) / total);
    }

    /** @return the alternate score preferred by this criterion
     *
     * @param eval			RegressionEvaluation object containing the scores
     * @param testingSet 	testing set with target values
     */
    protected double chooseAltScore(RegressionEvaluation eval, DataSet testingSet) {
        return eval.averageRSquared();
    }

    /**
     * @return the name of the tracked events for the relevant trainer
     */
    public String getEventsName() {
        return eventsName;
    }

    /**
     * @return the output from the lqst evaluation
     */
    public INDArray getOutput() {
        return output;
    }


    // SUBCLASSES

    /**
     * Subclass for saving the model with the best score.
     */
    public static class Score extends RunStats {

        public Score(MultiLayerNetwork model) {
            super(model);
        }

        /**
         * Check the model to see if we need to save this as the best model so far.
         *
         * @param model			current state of the model
         * @param testingSet	testing set for evaluating the model
         * @param processor		training processor managing the training
         * @param seconds		number of seconds spent processing this section
         * @param eventType		label to use for events
         * @param processType	label to use for processing
         */
        @Override
        public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor,
                double seconds, String eventType, String processType) {
            this.evaluateModel(model, testingSet, processor.getLabels());
            String saveFlag = "";
            if (this.newScore < this.getBestScore() || newScore == this.getBestScore() && this.newAccuracy > this.getBestAccuracy()) {
                this.setBestModel(model.clone());
                saveFlag = "  Model saved.";
            } else {
                saveFlag = String.format("  Best score was %g in %d with accuracy %g.", this.getBestScore(), this.getBestEvent(),
                        this.getBestAccuracy());
                this.uselessIteration();
            }
            log.info("Score after {} {} is {}. {} seconds to process {}. Accuracy = {}.{}",
                    this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
        }

        @Override
        public double getBestRating() {
            return -this.getBestScore();
        }

    }

    /**
     * Subclass for saving the model with the best accuracy.
     */
    public static class Accuracy extends RunStats {

        public Accuracy(MultiLayerNetwork model) {
            super(model);
        }

        /**
         * Check the model to see if we need to save this as the best model so far.
         *
         * @param model			current state of the model
         * @param testingSet	testing set for evaluating the model
         * @param processor		training processor managing the training
         * @param seconds		number of seconds spent processing this section
         * @param eventType		label to use for events
         * @param processType	label to use for processing
         */
        @Override
        public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor,
                double seconds, String eventType, String processType) {
            this.evaluateModel(model, testingSet, processor.getLabels());
            String saveFlag = "";
            if (this.newAccuracy > this.getBestAccuracy() || this.newAccuracy == this.getBestAccuracy() && this.newScore < this.getBestScore()) {
                this.setBestModel(model.clone());
                saveFlag = "  Model saved.";
            } else {
                saveFlag = String.format("  Best accuracy was %g in %d with score %g.", this.getBestAccuracy(),
                        this.getBestEvent(), this.getBestScore());
                this.uselessIteration();
            }
            log.info("Score after {} {} is {}. {} seconds to process {}. Accuracy = {}.{}",
                    this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
        }

    }

    /**
     * Subclass for training a regression model by score.
     */
    public static class Regression extends RunStats {

        protected Regression(MultiLayerNetwork model, RegressionTrainingProcessor processor) {
            super(model);
        }

        /**
         * Check the model to see if we need to save this as the best model so far.
         *
         * @param model			current state of the model
         * @param testingSet	testing set for evaluating the model
         * @param processor		training processor managing the training
         * @param seconds		number of seconds spent processing this section
         * @param eventType		label to use for events
         * @param processType	label to use for processing
         */
        @Override
        public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor, double seconds,
                String eventType, String processType) {
            this.scoreModel(model, testingSet, processor.getLabels());
            String saveFlag = "";
            if (this.newScore < this.getBestScore() || newScore == this.getBestScore() && this.newAccuracy > this.getBestAccuracy()) {
                this.setBestModel(model.clone());
                saveFlag = "  Model saved.";
            } else {
                saveFlag = String.format("  Best score was %g in %d with R-squared %g.", this.getBestScore(), this.getBestEvent(),
                        this.getBestAccuracy());
                this.uselessIteration();
            }
            log.info("Score after {} {} is {}. {} seconds to process {}. R-squared = {}.{}",
                    this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
        }

        @Override
        public double getBestRating() {
            return -this.getBestScore();
        }

    }

    /**
      * Subclass for training regression models by coefficient of determination.
      */
    public static class Coefficient extends RunStats {

        protected Coefficient(MultiLayerNetwork model, RegressionTrainingProcessor processor) {
            super(model);
        }

        /**
         * Check the model to see if we need to save this as the best model so far.
         *
         * @param model			current state of the model
         * @param testingSet	testing set for evaluating the model
         * @param processor		training processor managing the training
         * @param seconds		number of seconds spent processing this section
         * @param eventType		label to use for events
         * @param processType	label to use for processing
         */
        @Override
        public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor, double seconds,
                String eventType, String processType) {
            this.scoreModel(model, testingSet, processor.getLabels());
            String saveFlag = "";
            if (this.newAccuracy > this.getBestAccuracy() || newAccuracy == this.getBestAccuracy() && this.newScore < this.getBestScore()) {
                this.setBestModel(model.clone());
                saveFlag = "  Model saved.";
            } else {
                saveFlag = String.format("  Best determination coefficient was %g in %d with score %g.", this.getBestAccuracy(), this.getBestEvent(),
                        this.getBestScore());
                this.uselessIteration();
            }
            log.info("Score after {} {} is {}. {} seconds to process {}. Determination coefficient = {}.{}",
                    this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
        }

    }

    /**
     * Subclass for training regression models by Pearson correlation.
     */
   public static class Pearson extends RunStats {

       protected Pearson(MultiLayerNetwork model, RegressionTrainingProcessor processor) {
           super(model);
       }

       /**
        * Check the model to see if we need to save this as the best model so far.
        *
        * @param model			current state of the model
        * @param testingSet	testing set for evaluating the model
        * @param processor		training processor managing the training
        * @param seconds		number of seconds spent processing this section
        * @param eventType		label to use for events
        * @param processType	label to use for processing
        */
       @Override
       public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor, double seconds,
               String eventType, String processType) {
           this.scoreModel(model, testingSet, processor.getLabels());
           String saveFlag = "";
           if (this.newAccuracy > this.getBestAccuracy() || newAccuracy == this.getBestAccuracy() && this.newScore < this.getBestScore()) {
               this.setBestModel(model.clone());
               saveFlag = "  Model saved.";
           } else {
               saveFlag = String.format("  Best Pearson coefficient was %g in %d with score = %g.", this.getBestAccuracy(), this.getBestEvent(),
                       this.getBestScore());
               this.uselessIteration();
           }
           log.info("Score after {} {} is {}. {} seconds to process {}. Pearson coefficient = {}.{}",
                   this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
       }

       /**
        * @return the alternate score preferred by this criterion
        *
        * @param eval		RegressionEvaluation object containing the scores
        * @param testingSet	testing set containing the desired outcomes
        */
       protected double chooseAltScore(RegressionEvaluation eval, DataSet testingSet) {
           return eval.averagePearsonCorrelation();
       }

   }

   /**
    * Subclass for training regression models by pseudo-accuracy.
    */
  public static class PseudoAccuracy extends RunStats {

      private double bound;

      protected PseudoAccuracy(MultiLayerNetwork model, RegressionTrainingProcessor processor) {
          super(model);
          this.bound = processor.getBound();
      }

      /**
       * Check the model to see if we need to save this as the best model so far.
       *
       * @param model			current state of the model
       * @param testingSet	testing set for evaluating the model
       * @param processor		training processor managing the training
       * @param seconds		number of seconds spent processing this section
       * @param eventType		label to use for events
       * @param processType	label to use for processing
       */
      @Override
      public void checkModel(MultiLayerNetwork model, DataSet testingSet, LearningProcessor processor, double seconds,
              String eventType, String processType) {
          this.scoreModel(model, testingSet, processor.getLabels());
          String saveFlag = "";
          if (this.newAccuracy > this.getBestAccuracy() || newAccuracy == this.getBestAccuracy() && this.newScore < this.getBestScore()) {
              this.setBestModel(model.clone());
              saveFlag = "  Model saved.";
          } else {
              saveFlag = String.format("  Best accuracy was %g in %d with score = %g.", this.getBestAccuracy(), this.getBestEvent(),
                      this.getBestScore());
              this.uselessIteration();
          }
          log.info("Score after {} {} is {}. {} seconds to process {}. Accuracy = {}.{}",
                  this.getEventCount(), eventType, this.newScore, seconds, processType, this.newAccuracy, saveFlag);
      }

      /**
       * @return the alternate score preferred by this criterion
       *
       * @param eval		RegressionEvaluation object containing the scores
       * @param testingSet	testing set containing the desired outcomes
       */
      protected double chooseAltScore(RegressionEvaluation eval, DataSet testingSet) {
          return pseudoAccuracy(testingSet, bound);
      }

  }

}
