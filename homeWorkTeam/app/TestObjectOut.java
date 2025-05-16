package homeWorkTeam.app;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import homeWorkTeam.lib.Address;
public class TestObjectOut {
// 해시맵에 입력된애를 파일에 다가 입력해서 파일을 만든다. 오늘날짜도 함 붙여보고싶다.
	public static void main(String[] args) throws IOException { // 예외처리 안하고 테스트 빨리하려고 throws 해보았음..
		// 해시맵에 입력하는 방법 추가해봐야지~ k= 전번, v=Address  클래스
		HashMap<String, Address> myAddress = new HashMap<>(); // 얘는 주소 해시맵
		
		FileOutputStream fileOutputStream = null; //  try catch에 들어가 있으면 안된다.
		ObjectOutputStream objectOutputStream = null; //  try catch에 들어가 있으면 안된다.
		try {
			fileOutputStream = new FileOutputStream("c:\\temp\\test3.txt"); // 요 경로에 써줄애
			objectOutputStream = new ObjectOutputStream(fileOutputStream); // 얘가 내 입력을 받아서
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		myAddress.put("01011113333",new Address("cccc", "01011112222", "zzzz", "CCC")); //해시맵에 입력
		objectOutputStream.writeObject(myAddress); // 해시맵을 넣어봄 대충 이부분에서 해시맵에 입력하고 얘를 파일에 쓰면 되는거같다.
		
//		Address address1 = new Address("aaa", "01011112222", "zzzz", "AAA"); // 클래스에 정보입력
//		objectOutputStream.writeObject(address1); // 밖에있는 오브젝트에 쓰기
		
		objectOutputStream.close(); // 얘도 꼭 밖에 있어야됨..
		fileOutputStream.close();
	}
}
