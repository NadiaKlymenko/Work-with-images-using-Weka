import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * Created by nadia on 28.12.2015.
 */
public class PhotoOrDrawingClassifierTests {
    @Test
    public void testModel() throws Exception {
        PhotoOrDrawingClassifier classifier = new PhotoOrDrawingClassifier();
        classifier.train("D:\\Work\\Projects\\NadiasCoursework\\pdc2.arff");
        String filename;
        // a file containing paths to the test set images
        BufferedReader reader = new BufferedReader(
                new FileReader(new File("D:\\Work\\Projects\\NadiasCoursework\\test.txt")));
        while ((filename = reader.readLine()) != null) {
            String klass = classifier.test(filename);
            System.out.println(filename + " => " + klass);
        }
        reader.close();
    }
}
