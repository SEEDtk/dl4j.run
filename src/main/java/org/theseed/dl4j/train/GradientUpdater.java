/**
 *
 */
package org.theseed.dl4j.train;

import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;

/**
 * This is a utility class for creating the gradient updater.  Only static methods are provided.
 *
 * @author Bruce Parrello
 *
 */
public class GradientUpdater {

    public static enum Type {
        ADAM, NADAM, NESTEROVS, SGD
    }

    /**
     * @return an updater of the specified type with the specified learning rate
     *
     * @param type			type of updater
     * @param learningRate	learning rate parameter
     */
    public static IUpdater create(Type type, double learningRate) {
        IUpdater retVal = null;
        switch (type) {
        case ADAM :
            retVal = new Adam(learningRate);
            break;
        case NADAM :
            retVal = new Nadam(learningRate);
            break;
        case NESTEROVS :
            retVal = new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM);
            break;
        case SGD :
            retVal = new Sgd(learningRate);
            break;
        default :
            throw new IllegalArgumentException("Invalid updater type.");
        }
        return retVal;
    }

}
