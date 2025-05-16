package homeWorkTeam.app;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import homeWorkTeam.lib.Address;
public class TestObjectInput { // 파일 스트림 테스트
//	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException, ClassNotFoundException {
//		HashMap<String, Address> myAddress = new HashMap<>(); // 해시맵 준비
		// 얘는 필요없구냐.. 오브젝트스트림을 해시맵으로 강제형변환해준애를 그대로 쓰면 되기땜에. 따로 해시맵을 만들 필요가 없긴하다. 그럼.
		
		FileInputStream fileInputStream = null; // catch 밖에 써줄애 파일 읽어올 아이
		ObjectInputStream objectInputStream = null; // catch 밖에 있는애 오브젝트로 읽어올 아이
		
		fileInputStream = new FileInputStream("c:\\temp\\test3.txt"); // 얘도 예외처리 안하면 컴파일 오류
		objectInputStream = new ObjectInputStream(fileInputStream); // 얘도 예외처리 안하면 컴파일 오류
		
		
		HashMap<String, Address> myAddress = new HashMap<>(); // 내해시맵 여기에 readObjcet 해서 넣어줌
		myAddress = (HashMap)objectInputStream.readObject(); // 얘도 예외처리// 해시맵형태로 강제형변환해서 내해시맵에 대입하겠다.
		
		System.out.println(myAddress); // 해시맵이름을 넣을경우 배열처럼 출력
		System.out.println(myAddress.get("01011113333")); // 미리 입력해둔 키를 그냥 넣어봄.
	
		objectInputStream.close();
		fileInputStream.close();
	
	}
}
