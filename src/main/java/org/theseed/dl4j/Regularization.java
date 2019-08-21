/**
 *
 */
package org.theseed.dl4j;

import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;

/**
 * This object indicates the type of regularization. It provides methods for applying the regulizer to a builder
 * and for displaying the regularization style.
 *
 * @author Bruce Parrello
 *
 */
public class Regularization {

    public static enum Mode {
        /** use gaussian dropout */
        GAUSS,
        /** use linear dropout */
        LINEAR,
        /** use L2 regulation */
        L2,
        /** weight decay */
        WEIGHT_DECAY,
        /** do not regularize */
        NONE
    }

    // FIELDS
    /** mode of regularization */
    private Mode mode;
    /** regularization factor */
    private double factor;

    /**
     * Create the regularization object.
     *
     * @param mode		mode of regularization
     * @param factor	factor-- higher is more sensitive, lower is less
     */
    public Regularization(Mode mode, double factor) {
        this.mode = mode;
        this.factor = factor;
    }

    /**
     * Apply this object to a builder.  We have one of these for each builder type because there is
     * no common subclass, which is insane.
     *
     * @param builder	builder for the current layer
     */
    public void apply(DenseLayer.Builder builder) {
        switch (this.mode) {
        case GAUSS :
            builder.dropOut(new GaussianDropout(this.factor));
            break;
        case LINEAR :
            builder.dropOut(new Dropout(this.factor));
            break;
        case L2 :
            builder.l2(this.factor);
            break;
        case WEIGHT_DECAY :
            builder.weightDecay(this.factor);
            break;
        case NONE :
            break;
        }
    }

    /**
     * Apply this object to a builder.  We have one of these for each builder type because there is
     * no common subclass, which is insane.
     *
     * @param builder	builder for the current layer
     */
    public void apply(LSTM.Builder builder) {
        switch (this.mode) {
        case GAUSS :
            builder.dropOut(new GaussianDropout(this.factor));
            break;
        case LINEAR :
            builder.dropOut(new Dropout(this.factor));
            break;
        case L2 :
            builder.l2(this.factor);
            break;
        case WEIGHT_DECAY :
            builder.weightDecay(this.factor);
            break;
        case NONE :
            break;
        }
    }

    /**
     * @return a string describing the regularization protocol
     */
    public String toString() {
        String retVal = null;
        switch (this.mode) {
        case GAUSS :
        case LINEAR :
            retVal = String.format("%s dropout with factor %g", this.mode, this.factor);
            break;
        case L2 :
        case WEIGHT_DECAY :
            retVal = String.format("%s with coefficient %g", this.mode, this.factor);
            break;
        case NONE :
            retVal = "NONE";
        }
        return retVal;
    }


}
