package homeWorkTeam.app;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

import homeWorkTeam.lib.Address;
public class AddressStreamFirtstTry {
	public static void main(String[] args) {
//		키보드 입력도 준비..
		Scanner scanner = new Scanner(System.in);
//		메뉴1~5입력할 input // 내클래스에 입력받을 변수 4개 :  이름, 전화, 주소, 그룹(구분)
		String input = null;
		
		String inputName = null;
		String inputPhone = null;
		String inputHome = null;
		String inputGroup = null;
//		해시맵 준비 <전화번호, 내클래스>
		HashMap<String, Address> myAddress = new HashMap<>();

		
//		파일 읽어오기 // 로딩시 1회만 읽어서 해시맵에 대입하고 닫는다. 
		FileInputStream fileInputStream = null;
		ObjectInputStream objectInputStream = null;
		
		try {
			// 파일을 저장경로에서 읽어올 스트림 만들기 fileInput->objectInput
			fileInputStream = new FileInputStream("c:\\temp\\myAddressMap.txt");
			objectInputStream = new ObjectInputStream(fileInputStream);
			// 내해시맵에 readObject()를 해시맵으로 형변환해서 대입한다.
			// 여기서 대입됬다면 해시맵만 가지고 작업할꺼니 스트림은 다 닫아도 될것같다.
			myAddress = (HashMap)objectInputStream.readObject();
//			System.out.println(myAddress); //닫기전에 출력할생각인데 이거 비어있을수도 있다. 테스트
		} catch (FileNotFoundException e) {
			System.out.println("파일이 없으니 이번에 새로 만들자 ^ㅁ^");
//			System.out.println(myAddress); // 빈 배열이 출력된다. 빈 배열은 존재하긴 함 ^ㅁ^
		} catch (IOException e) {
			System.out.println("여기서도 오류인가 oTL");
		} catch (ClassNotFoundException e) {
			System.out.println("필요한 클래스 import했는지 봐라~ ㅇ0ㅇ;");
		} finally { // 읽기 스트림 닫기
			try {	// 만약 파일Input스트림이 제대로 입력됬다면~ 두개 스트림 다 닫아준다.
				if(fileInputStream !=null) {
					fileInputStream.close();
					objectInputStream.close();
				}
			} catch (IOException e) {
			}
		}// try catch finally // 스트림 다 닫음
		
		
		while(true) { // 초기화면
			
			System.out.println("===================================");
			System.out.println("   다음 메뉴 중 하나를 선택하세요.");
			System.out.println("===================================");
			System.out.println("1. 회원추가");
			System.out.println("2. 회원 목록 보기");
			System.out.println("3. 회원 정보 수정하기");
			System.out.println("4. 회원 삭제");
			System.out.println("5. 종료");
			System.out.println("===================================");
			System.out.println();
			System.out.print("번호 선택>> ");
			input= scanner.nextLine(); // 메뉴 입력 받음
			
			if(!(input.equals("1")||input.equals("2")||
					 input.equals("3")||input.equals("4")||input.equals("5"))) {
				System.out.println("다시 입력해주세요.");
			} // 1~5 아닐경우 처음으로 돌아가고 재입력 요청.. 아니면 if를 계속 써줘야된다. while이라 상관없을거같다.
			if(input.equals("5")) {
				System.out.println("종료되었습니다.");
				break; // App종료 = while 나가기
			}
			switch(input) { // 메뉴선택
			case "1":
				System.out.println();
				System.out.println("등록할 회원의 정보를 입력하세요");
				System.out.print("이름: ");
				inputName = scanner.nextLine(); // 1 이름 입력
				
				boolean flag = false; // do밖에 있어야 되나보다. while에 안닿나보다.
				do {
					System.out.print("전화번호(ex:01011112222) : ");
					inputPhone = scanner.nextLine();
					flag = false; // 밖에만 있다면 true로 걸렸을때 바뀌질 않는다. 
					// 재입력 조건에 걸리지 않는다면 입력만 받아서 즉시탈출 가능함
					for(char phone:inputPhone.toCharArray()) {
						if(!(Character.isDigit(phone))) {
							System.out.println("전화번호를 다시 입력하세요.");
							flag = true;
							break; //문자가 하나라도 걸리면 나온다. 
						}
						if(inputPhone.length()!=11) {
							System.out.println("올바른 전화번호 형식이 아닙니다.");
							flag = true;
							break; 
						} // 전화번호 길이 검증
					}// isDigit? 문자 검증
					//System.out.println(flag); // flag 상태 확인
				}while(flag); // 전화번호 입력 검증
				
				System.out.print("주소: ");
				inputHome = scanner.nextLine(); // 주소 입력
				
				do {
					flag = false;
					System.out.print("구분(a가족/b친구/c기타 중에 선택): ");
					inputGroup = scanner.nextLine();//if에 걸리지 않으면 즉시 탈출
					if(!(inputGroup.equals("a")||inputGroup.equals("b")||inputGroup.equals("c"))) {
						System.out.println("a,b,c중 하나를 입력하세요.");
						flag = true; 
					}
					switch(inputGroup) {
					case "a":
						inputGroup = "가족";
						break;
					case "b":
						inputGroup = "친구";
						break;
					case "c":
						inputGroup = "기타";
						break;
					} // 분류, 그룹입력
				}while(flag); // 분류(group) 검증 입력
				
				// 해시맵에 입력 put()
				myAddress.put(inputPhone, new Address(inputName,inputPhone,inputHome,inputGroup));
				
				FileOutputStream fileOutputStream = null; // 파일에 쓰기 준비
				ObjectOutputStream objectOutputStream = null;
				// 파일에 쓰기
				try {
					fileOutputStream = new FileOutputStream("c:\\temp\\myAddressMap.txt");
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
				break; // switch만 나간다 --> 처음으로 돌아간다. 
			case "2":
				System.out.println();
				System.out.println("총"+ myAddress.size() +"명의 회원이 저장되어 있습니다.");

				Set<String> keys = myAddress.keySet();	//얘들이 바깥에 있으면 왜인지 출력이 안된다.
				Iterator<String> iter = keys.iterator(); //iterator는 hasNext 하면 1회용! 출력시마다 만들어보자.
				int i = 1; // 출력시 항상 번호 붙여주고 싶을때
				while(iter.hasNext()) {
					String key = iter.next();
					System.out.println(i + "." + myAddress.get(key));
					i++;
				}
				String noMeaning = scanner.nextLine(); // 목록 확인하라고 잠시 멈춰줬음
				break;
			case "3":
				System.out.println("3 입력됨");
				break;
			case "4":
				System.out.println("4 입력됨");
				break;
			} //switch
			
			
			
			
		} // while(true) 초기화면
	scanner.close();	
	} //main end
} //class end





