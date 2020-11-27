/**
 *
 */
package org.theseed.dl4j;

import org.theseed.utils.ICommand;

/**
 * This class performs cross-validation on a regression training set.  The set is divided into equal portions (folds)
 * and a model is trained once for each fold, with the fold used as the testing set and the remaining data used for
 * testing.  Each model is then evaluated against the entire training set and the best model is saved.
 *
 * To determine the rating of the model, the mean average error is computed for each output label as a percent of
 * the value range, and
 *
 * The positional parameters are the name of the parameter file and the name of the model directory.
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -v	show more detailed progress messages
 * -k	number of folds to use (default 10)
 *
 * @author Bruce Parrello
 *
 */
public class CrossValidateProcessor implements ICommand {

    // FIELDS
    // TODO data members for CrossValidateProcessor

    @Override
    public boolean parseCommand(String[] args) {
        // TODO code for parseCommand
        return false;
    }

    @Override
    public void run() {
        // TODO code for run

    }

}
