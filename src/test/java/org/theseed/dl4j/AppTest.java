package org.theseed.dl4j;

import junit.framework.Test;

import junit.framework.TestCase;
import junit.framework.TestSuite;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Unit test for simple App.
 */
public class AppTest extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public AppTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( AppTest.class );
    }

    /**
     * Test the training set reader.
     * @throws IOException
     */
    public void testReader() throws IOException {
        File inFile = new File("src/test", "iris.tbl");
        List<String> labels = Arrays.asList("setosa", "versicolor", "virginica");
        TabbedDataSetReader reader = new TabbedDataSetReader(inFile, "species", labels)
                .setBatchSize(11);
        assertTrue("End of file too soon.", reader.hasNext());
        assertThat("Wrong input width in file.", reader.getWidth(), equalTo(4));
        DataSet set1 = reader.next();
        assertThat("Wrong number of inputs.", set1.numInputs(), equalTo(4));
        assertThat("Wrong number of classes.", set1.numOutcomes(), equalTo(3));
        assertThat("Wrong batch size.", set1.numExamples(), equalTo(11));
        INDArray features = set1.getFeatures();
        INDArray outputs = set1.getLabels();
        assertThat("Wrong sl for ex1.", features.getDouble(0, 0), closeTo(6.0, 0.001));
        assertThat("Wrong sw for ex1.", features.getDouble(0, 1), closeTo(3.4, 0.001));
        assertThat("Wrong pl for ex1.", features.getDouble(0, 2), closeTo(4.5, 0.001));
        assertThat("Wrong pw for ex1.", features.getDouble(0, 3), closeTo(1.6, 0.001));
        assertThat("Wrong label for ex1.", outputs.getDouble(0, 0), equalTo(0.0));
        assertThat("Wrong label aura 1 for ex1.", outputs.getDouble(0, 1), equalTo(1.0));
        assertThat("Wrong label aura 2 for ex1.", outputs.getDouble(0, 2), equalTo(0.0));
        assertThat("Wrong sl for ex9.", features.getDouble(8, 0), closeTo(5.2, 0.001));
        assertThat("Wrong sw for ex9.", features.getDouble(8, 1), closeTo(4.1, 0.001));
        assertThat("Wrong pl for ex9.", features.getDouble(8, 2), closeTo(1.5, 0.001));
        assertThat("Wrong pw for ex9.", features.getDouble(8, 3), closeTo(0.1, 0.001));
        assertThat("Wrong label for ex9.", outputs.getDouble(8, 0), equalTo(1.0));
        assertThat("Wrong label aura 1 for ex9.", outputs.getDouble(8, 1), equalTo(0.0));
        assertThat("Wrong label aura 2 for ex9.", outputs.getDouble(8, 2), equalTo(0.0));
        int count = 1;
        for (DataSet seti : reader) {
            count++;
            if (count <= 13) {
                assertThat("Wrong batch size for " + count, seti.numExamples(), equalTo(11));
            } else {
                assertThat("Wrong tail batch size", seti.numExamples(), equalTo(7));
                assertThat("Wrong label for last example.", seti.getLabels().getDouble(6, 2),
                        equalTo(1.0));
            }
        }
        inFile = new File("src/test", "samples.tbl");
        reader = new TabbedDataSetReader(inFile, "lcol", labels, Arrays.asList("mcol", "mcol2"));
        set1 = reader.next();
        assertThat("Wrong number of inputs.", set1.numInputs(), equalTo(3));
        assertThat("Wrong number of classes.", set1.numOutcomes(), equalTo(3));
        assertThat("Wrong batch size.", set1.numExamples(), equalTo(2));
        features = set1.getFeatures();
        outputs = set1.getLabels();
        assertThat("Wrong col1 for ex1.", features.getDouble(0, 0), closeTo(1.0, 0.001));
        assertThat("Wrong col2 for ex1.", features.getDouble(0, 1), closeTo(3.0, 0.001));
        assertThat("Wrong col3 for ex1.", features.getDouble(0, 2), closeTo(6.0, 0.001));
        assertThat("Wrong label for ex1.", outputs.getDouble(0, 0), equalTo(1.0));
        assertThat("Wrong label aura 1 for ex1.", outputs.getDouble(0, 1), equalTo(0.0));
        assertThat("Wrong label aura 2 for ex1.", outputs.getDouble(0, 2), equalTo(0.0));
        assertThat("Wrong col1 for ex2.", features.getDouble(1, 0), closeTo(1.1, 0.001));
        assertThat("Wrong col2 for ex2.", features.getDouble(1, 1), closeTo(3.1, 0.001));
        assertThat("Wrong col3 for ex2.", features.getDouble(1, 2), closeTo(6.1, 0.001));
        assertThat("Wrong label for ex2.", outputs.getDouble(1, 2), equalTo(1.0));
        assertThat("Wrong label aura 1 for ex2.", outputs.getDouble(1, 0), equalTo(0.0));
        assertThat("Wrong label aura 2 for ex2.", outputs.getDouble(1, 1), equalTo(0.0));
        inFile = new File("src/test", "predicts.tbl");
        reader = new TabbedDataSetReader(inFile, Arrays.asList("mcol", "mcol2"));
        set1 = reader.next();
        features = set1.getFeatures();
        List<String> metaData = set1.getExampleMetaData(String.class);
        assertThat("Wrong col1 for ex1.", features.getDouble(0, 0), closeTo(1.0, 0.001));
        assertThat("Wrong col2 for ex1.", features.getDouble(0, 1), closeTo(3.0, 0.001));
        assertThat("Wrong col3 for ex1.", features.getDouble(0, 2), closeTo(6.0, 0.001));
        assertThat("Wrong meta for ex1.", metaData.get(0), equalTo("2.0\t5.0"));
        assertThat("Wrong col1 for ex2.", features.getDouble(1, 0), closeTo(1.1, 0.001));
        assertThat("Wrong col2 for ex2.", features.getDouble(1, 1), closeTo(3.1, 0.001));
        assertThat("Wrong col3 for ex2.", features.getDouble(1, 2), closeTo(6.1, 0.001));
        assertThat("Wrong label for ex2.", metaData.get(1), equalTo("2.1\t5.1"));
    }


}
