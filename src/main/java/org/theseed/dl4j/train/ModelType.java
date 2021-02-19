/**
 *
 */
package org.theseed.dl4j.train;

import org.theseed.utils.IDescribable;

/**
 * This defines the type of a model.  The various model types have a lot of things in common, but they also
 * differ greatly.  The model type is the basic starting point for creating common tools for manipulating AI
 * models.
 *
 * The three current types are
 *
 * 	REGRESSION		neural net with one or more real-number outputs
 *
 *  CLASS			neural net with a single discreet output
 *
 *  DECISION		random-forest classifier with a single discreet output
 *
 * @author Bruce Parrello
 *
 */
public enum ModelType implements IDescribable {
    REGRESSION {
        @Override
        public Enum<?>[] getPreferTypes() {
            return RunStats.RegressionType.values();
        }
        @Override
        public String getDescription() {
            return "Regression";
        }
        @Override
        public String getDisplayType() {
            return "ScatterGraph";
        }
        @Override
        public String resultDescription() {
            return "Scatter Graph";
        }
        @Override
        public int metaLabel() {
            return 0;
        }
    }, CLASS {
        @Override
        public Enum<?>[] getPreferTypes() {
            return RunStats.OptimizationType.values();
        }
        @Override
        public String getDescription() {
            return "Classification";
        }
        @Override
        public String getDisplayType() {
            return "ConfusionMatrix";
        }
        @Override
        public String resultDescription() {
            return "Confusion Matrix";
        }
        @Override
        public int metaLabel() {
            return 1;
        }
    }, DECISION {

        @Override
        public String getDescription() {
            return "RandomForest";
        }

        @Override
        public Enum<?>[] getPreferTypes() {
            return null;
        }

        @Override
        public String getDisplayType() {
            return "ConfusionMatrix";
        }

        @Override
        public String resultDescription() {
            return "Confusion Matrix";
        }

        @Override
        public int metaLabel() {
            return 1;
        }

    };

    /**
     * @return the available preference types for models of this type (neural nets only)
     */
    public abstract Enum<?>[] getPreferTypes();

    /**
     * @return the result display for models of this type
     */
    public abstract String getDisplayType();

    /**
     * @return the description for the result display of models of this type
     */
    public abstract String resultDescription();

    /**
     * @return 1 if this model type has a classification label column, else 0
     */
    public abstract int metaLabel();

    /**
     * @return a training processor of the specified type
     *
     * @param type	type of training-- regressions or classification
     */
    public static ITrainingProcessor create(ModelType type) {
        ITrainingProcessor retVal = null;
        switch (type) {
        case REGRESSION :
            retVal = new RegressionTrainingProcessor();
            break;
        case CLASS :
            retVal = new ClassTrainingProcessor();
            break;
        case DECISION :
            retVal = new RandomForestTrainProcessor();
            break;
        }
        return retVal;
    }

}
