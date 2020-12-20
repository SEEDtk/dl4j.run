/**
 *
 */
package org.theseed.dl4j.train;

import java.util.Iterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;

/**
 * This is an abstract base class for a trainer.  Trainers are used by processors to train a model and return
 * run statistics.
 *
 * @author Bruce Parrello
 *
 */
public abstract class Trainer {

    // FIELDS
    /** relevant training processor */
    protected LearningProcessor processor;
    /** logger for messages */
    protected Logger log;

    /**
     *	Create a new trainer for a specified training processor.
     *
     * @param processor		the training processor that created the model
     * @param log			logging facility for messages
     */
    public Trainer(LearningProcessor processor, Logger log) {
        this.processor = processor;
        this.log = log;
    }

    /**
     * Abstract method to train a model.
     *
     * @param model			the model to train
     * @param reader		the reader to read in the dataset
     * @param testingSet	testing set for early termination
     * @param runStats		a RunStats for the training operation
     * @param progressMonitor
     *
     * @throws InterruptedException
     */
    public abstract void trainModel(MultiLayerNetwork model, Iterator<DataSet> reader,
            DataSet testingSet, RunStats runStats, ITrainReporter progressMonitor) throws InterruptedException;

    /**
     * @return the plural name for an event in this trainer's cycle
     */
    public abstract String eventsName();

    /** types of trainers */
    public enum Type { BATCH, EPOCH }

    /**
     * @return a trainer of the specified type.
     *
     * @param type			type of trainer to create
     * @param processor		the training processor that created the model
     * @param log			logging facility for messages
     */
    public static Trainer create(Type trainerType, LearningProcessor processor, Logger log) {
        Trainer retVal = null;
        switch (trainerType) {
        case BATCH :
            retVal = new BatchTrainer(processor, log);
            break;
        case EPOCH :
            retVal = new EpochTrainer(processor, log);
            break;
        default :
            throw new IllegalArgumentException("Invalid trainer type.");
        }
        return retVal;
    }


}
