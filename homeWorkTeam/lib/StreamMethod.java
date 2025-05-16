package homeWorkTeam.lib;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;

public class StreamMethod { // 파일 읽기 쓰기 클래스

	//	HashMap<String, Address> myAddress = new HashMap<>(); // 해시맵을 파라미터에 선언한게되서 따로 선언이 필요없었다
	// 	파일 읽어오기 // 로딩시 1회만 읽어서 해시맵에 대입하고 닫는다. 
	public HashMap<String, Address> inputStream(HashMap<String, Address> myAddress) {
		FileInputStream fileInputStream = null;
		ObjectInputStream objectInputStream = null;
		
		try {
			// 파일을 저장경로에서 읽어올 스트림 만들기 fileInput->objectInput
			fileInputStream = new FileInputStream("c:\\temp\\myAddressMapTest.txt");
			objectInputStream = new ObjectInputStream(fileInputStream);
			// 내해시맵에 형변환해서 대입한다.
			myAddress = (HashMap)objectInputStream.readObject();
			
		} catch (FileNotFoundException e) {
			System.out.println("파일이 없으니 이번에 새로 만들죠~ ^ㅁ^");
		} catch (IOException e) {
			System.out.println("입출력 오류입니다~ oTL");
		} catch (ClassNotFoundException e) {
			System.out.println("필요한 클래스 import했는지 보세요~ ㅇ0ㅇ;");
		} finally { // 읽기 스트림 닫기
			try {	// 만약 파일Input스트림이 제대로 입력됬다면~ 두개 스트림 다 닫아준다.
				if(fileInputStream !=null) {
					fileInputStream.close();
					objectInputStream.close();
				}
			} catch (IOException e) {
			}
		}// try catch finally // 스트림 다 닫음
		return myAddress;
	} //inputStream 파일 읽어오기 메소드
	
	//	파일 쓰기 + 해시맵에 입력 // 리턴할 내용 없으므로 void   
	public void outputStream(String inputName, String inputPhone, String inputHome, String inputGroup,
			HashMap<String, Address> myAddress) {
		// 해시맵에 입력
		myAddress.put(inputPhone, new Address(inputName,inputPhone,inputHome,inputGroup));
		// 파일에 쓰기 스트림 준비
		FileOutputStream fileOutputStream = null; 
		ObjectOutputStream objectOutputStream = null;
		// 파일에 쓰기
		try {
			fileOutputStream = new FileOutputStream("c:\\temp\\myAddressMapTest.txt");
			objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(myAddress); // 예외발생이 안되면 여기서 파일 쓰기 
		} catch (FileNotFoundException e) {
			System.out.println("파일을 쓸 수 있는 디렉토리인지 확인하세요.");
		} catch (IOException e) { // IO 입출력 예외
			System.out.println("입출력 오류입니다.");
		}finally {
			try {
				objectOutputStream.close();
				fileOutputStream.close();
			} catch (IOException e) {
			}
		}// try catch 파일 쓰기, 닫기
	} // outputStream 파일에 쓰기 메소드
	
} // class
