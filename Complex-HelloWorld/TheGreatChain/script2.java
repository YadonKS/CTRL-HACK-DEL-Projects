import java.io.*;

public class script2 {
    public static void main(String[] args) {
        System.out.println("In Java, moving to Batch script...");
        try {
            ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "script3.bat");
            pb.inheritIO().start().waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
