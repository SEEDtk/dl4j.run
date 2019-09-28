package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
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


    /** Method for determining model to save */
    public static enum OptimizationType {
        /** save the model with the lowest score */
        SCORE,
        /** save the model with the highest accuracy */
        ACCURACY;
    }

    /** logging facility */
    public static Logger log = LoggerFactory.getLogger(TrainingProcessor.class);

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
    private INDArray output;

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
        this.bestAccuracy = 0;
        this.uselessEvents = 0;
        this.bestScore = Double.MAX_VALUE;
        this.duration = "00:00";
    }

    /**
     * Construct a run tracker of the appropriate type.
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
     * @param logDir	directory containing the trial log
     * @param label		heading comment, if any
     * @param report	text of the report to write, with internal new-lines
     *
     * @throws IOException
     */
    public static void writeTrialReport(File logDir, String label, String report) throws IOException {
        // Open the trials log in append mode and write the information about this run.
        File trials = new File(logDir, "trials.log");
        PrintWriter trialWriter = new PrintWriter(new FileWriter(trials, true));
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

}
