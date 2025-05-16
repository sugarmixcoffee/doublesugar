package homeWorkTeam.lib;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
public class AddressStream { // 파일에서 불러오기 // 파일에 저장하기
	
	// 파라미터에 이미 데이터 타입을 적어줘서 myAddress는 따로 선언해줄 필요가 없다
	// 파일 읽어오기 // 시작할때 파일 내용을 해시맵에 담아온다.
	public HashMap<String, Address> inputStream(HashMap<String, Address> myAddress) {
		FileInputStream fileInputStream = null;
		ObjectInputStream objectInputStream = null;
		
		try {
			// 파일을 저장경로에서 읽어올 스트림 만들기 fileInput->objectInput
			fileInputStream = new FileInputStream("c:\\temp\\myAddressMapTest.txt");
			objectInputStream = new ObjectInputStream(fileInputStream);
			// 내해시맵에 readObject()를 해시맵으로 형변환해서 대입한다. 읽어오기 스트림은 닫는다.
			myAddress = (HashMap)objectInputStream.readObject();
			
		} catch (FileNotFoundException e) {
			System.out.println("파일이 없으니 이번에 새로 만들자 ^ㅁ^");
			//System.out.println(myAddress); // 빈 배열이 출력된다. 테스트 용
		} catch (IOException e) {// IO 입출력 예외
			System.out.println("여기서도 오류인가 oTL");
		} catch (ClassNotFoundException e) { // 클래스 import 안했을 경우
			System.out.println("필요한 클래스 import했는지 봐라~ ㅇ0ㅇ;");
		} finally { // 읽기 스트림 닫기
			try {	// 요까지 오면 파일Input스트림이 제대로 입력된것 
				if(fileInputStream !=null) {
					fileInputStream.close();
					objectInputStream.close();
				}
			} catch (IOException e) {
			}
		} // try catch finally // 스트림 다 닫음
		return myAddress;
	} // inputStream 파일 읽어오기
	
	// 입력 내용을 받아서 해시맵 + 파일 쓰기. 리턴할 내용 없으므로 void // main에서 선언해줄생각이라서 static 없음
	public void outputStream(String inputName, String inputPhone, String inputHome, String inputGroup,
			HashMap<String, Address> myAddress) {
		
		// 해시맵에 써주기
		myAddress.put(inputPhone, new Address(inputName,inputPhone,inputHome,inputGroup));
		
		// 파일에 쓰기 메소드사용
		makeAddressFile(myAddress);
	} // outputStream 파일 읽어오기
	
	// 입력없이, 해시맵만 파일에 쓰기
	// makeAddressFile
	public void makeAddressFile(HashMap<String, Address> myAddress) {
		FileOutputStream fileOutputStream = null; // 파일에 쓸 스트림 준비
		ObjectOutputStream objectOutputStream = null;
		
		// 파일에 쓰기
		try {
			fileOutputStream = new FileOutputStream("c:\\temp\\myAddressMapTest.txt");
			objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(myAddress); // 예외발생이 안되면 여기서 파일 쓰기 되야함
		} catch (FileNotFoundException e) {
			System.out.println("파일을 쓸수 없는 디렉토리인 경우");
		} catch (IOException e) { // IO 입출력 예외
			System.out.println("입출력 오류 경우");
		}finally {
			try {
				objectOutputStream.close();
				fileOutputStream.close();
			} catch (IOException e) {
			}
		}// try catch 파일 쓰기, 닫기
	} //makeAddressFile 파일 쓰기
	
	// 생성자를 쓰면 메소드 이름이 같아지니까. 입출력시 헷갈려서.. 일반 메소드 사용함
}

