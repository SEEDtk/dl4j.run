/**
 *
 */
package org.theseed.dl4j.train;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.theseed.utils.Parms;

/**
 * @author Bruce Parrello
 *
 */
class TestTrainingSetup {

    @Test
    void testModelSetup() throws IOException {
        ITrainingProcessor processor = ModelType.create(ModelType.DECISION);
        File modelDir = new File("src/test/data", "RnaSeqWide");
        List<String> possibles = getPossibles(processor, modelDir);
        assertThat(possibles, containsInAnyOrder("prod_level", "production", "growth"));
        processor = ModelType.create(ModelType.REGRESSION);
        modelDir = new File("src/test/data", "Threonine");
        possibles = getPossibles(processor, modelDir);
        assertThat(possibles, containsInAnyOrder("production", "density"));
    }

    /**
     * Compute the possible columns to select for the training heat map.
     *
     * @param processor		model processor
     * @param modelDir		model directory
     *
     * @return a list of the column names
     *
     * @throws IOException
     */
    public List<String> getPossibles(ITrainingProcessor processor, File modelDir) throws IOException {
        File parmFile = new File(modelDir, "parms.prm");
        Parms parms = new Parms(parmFile);
        assertThat(processor.setupParameters(parms, modelDir), equalTo(true));
        List<String> retVal = new ArrayList<String>(processor.getMetaList());
        retVal.addAll(processor.getLabelCols());
        retVal.remove(processor.getIdCol());
        return retVal;
    }

}
