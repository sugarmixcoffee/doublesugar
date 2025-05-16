package homeWorkTeam.lib;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

public class AddressMethodClub2 {
	// 반복문 탈출 등에 사용할 깃발
	private boolean flag;
	// 연락처 추가 메소드 (해시맵 입력+ 파일쓰기) inputAddress(scanner, myAddress, myStreamMethod)
	public void inputAddress(Scanner scanner, HashMap<String, Address> myAddress, AddressStream myStreamMethod) {
		String inputName = null; // 입력할때만 쓰이므로 이 메소드 내에서만 사용
		String inputPhone = null;
		String inputHome = null;
		String inputGroup = null;
		
		System.out.println();
		System.out.println("등록할 회원의 정보를 입력하세요");
		System.out.print("이름: ");
		inputName = scanner.nextLine(); // 이름 입력 받기
		do {
			System.out.print("전화번호(ex:01011112222) : ");
			inputPhone = scanner.nextLine();
			flag = false; // 재입력 조건에 걸리지 않는다면 입력만 받고 즉시탈출 
			 
			for (char phone : inputPhone.toCharArray()) {
				if (!(Character.isDigit(phone))) {
					System.out.println("전화번호를 다시 입력하세요.");
					flag = true;
					break; // 문자가 발견되면 처음으로 돌아감
				}
				if (inputPhone.length() != 11) {
					System.out.println("올바른 전화번호 형식이 아닙니다.");
					flag = true;
					break;
				} // 길이 검증
			} // 전화번호 입력 검증 for
		} while (flag); // 전화번호 입력 검증 do while
		
		System.out.print("주소: ");
		inputHome = scanner.nextLine(); // 주소 입력
		do {
			flag = false; // if에 걸리지 않으면 즉시탈출
			System.out.print("구분(a가족/b친구/c기타 중에 선택): ");
			inputGroup = scanner.nextLine();
			if (!(inputGroup.equals("a") || inputGroup.equals("b") || inputGroup.equals("c"))) {
				System.out.println("a,b,c중 하나를 입력하세요.");
				flag = true; // 선택문자 외 입력시 재입력 요청
			}
			switch (inputGroup) {
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
		} while (flag); // 분류(group) 검증 입력
		
		// 아래 메소드에 4개 넣어줘야 하므로 여기까지 메소드 포함시켜둠
		// 파일 쓰기용 메소드 :  변수(해시맵에 써넣을 재료) + 해시맵(app에서 사용할 맵 + 파일에 쓰기)
		myStreamMethod.outputStream(inputName, inputPhone, inputHome, inputGroup, myAddress);
		
		// 저장만 하면 되니까 리턴 없어도 된다. 
	} // inputAddress
	
	//viewAllAddress // 목록 출력 메소드
	public void viewAllAddress(HashMap<String, Address> myAddress) {
		// iterator는 hasNext 하면 1회용! 메소드 실행시마다 만든다.
		Set<String> keys = myAddress.keySet();
		Iterator<String> iter = keys.iterator();
		System.out.println();
		System.out.println("총 " + myAddress.size() + "명의 회원이 저장되어 있습니다.");
		int num = 1; // 출력시 항상 번호 붙여주고 싶을때
		while (iter.hasNext()) {
			String key = iter.next();
			System.out.println(num + "." + myAddress.get(key));
			num++;
		}
	} // viewAllAddress
	
	// 검색+수정 메소드 (검색한 사람의 키=전화번호) serchAddress(scanner, myAddress) 
	public String serchAddress(Scanner scanner, HashMap<String, Address> myAddress,String input) {
		// 해시맵에서 k를 차례로 꺼내기 위한 iterator 준비
		Set<String> keys = myAddress.keySet();
		Iterator<String> iter = keys.iterator();
		// 검색내용이 없다면 false, '없음'메시지가 출력된다.
		flag = false;
		
		// input을 메인에서 받아와서 / 3일 경우->수정메시지 출력 / 4일 경우 --> 삭제 메세지 출력 
		if(input.equals("3")) {
			System.out.print("수정할 회원의 이름을 입력하세요: ");
		}if(input.equals("4")) {
			System.out.print("삭제할 회원의 이름을 입력하세요: ");
		}
		String inputEdit = scanner.nextLine();
		// 검색 내용이 여러개일경우 선택을 위해서 붙여줄 번호
		int listNum = 0;
		// 전화번호를 임시 보관 array. 검색에 걸릴때마다 기록해둘 생각임
		ArrayList<String> arrList = new ArrayList<String>();

		while (iter.hasNext()) { // 이름 검색하는 반복문
			String key = iter.next();
			if (inputEdit.equals(myAddress.get(key).getName())) { // 주소록 클래스에서 이름을 빼와서 대조해본다.
				listNum++;
				// 검색에 걸린다면. v값(전체 정보) 출력해주게 됨
				System.out.println(listNum + "." + myAddress.get(key));
				arrList.add(key); // 출력할때 전화번호 차례로 저장. 인덱스=listNum-1
				flag = true; // 회원 목록이 있다면 true로 바꿔서 오류메세지가 나오지 않게 해준다.
			}
		} // 이름 검색 while hasNext

		if (flag == false) { // 검색 안되면 false. case 3 나가기. 리턴 null 로 지정해서 main의 반복문 처음으로 돌아가도록 함
			throw new RuntimeException("해당하는 회원 정보가 없습니다.");
//			System.out.println("해당하는 회원 정보가 없습니다.");
//			return null; // 스위치문에서는 break 사용했었음
		}

		// 검색해서 목록이 출력된 경우
		System.out.println("총 " + listNum + "개의 목록이 검색되었습니다.");
		int inputNum = 0;

		do {
			flag = false;
			if(input.equals("3")) {
				System.out.print("수정할 목록 선택 : ");
			} else if(input.equals("4")) {
				System.out.print("삭제할 목록 선택 : ");
			}
			// 목록 숫자 입력받으면서 혹시 있을 공백도 제거한다
			String selectNum = scanner.nextLine().replaceAll(" ", ""); 

			try {
				inputNum = Integer.parseInt(selectNum); // 숫자로 강제 형변환해서 확인
				if ((inputNum <= 0) || (inputNum > listNum)) {  // 검색된 리스트를 선택하지 않는다면 반복
					System.out.println("잘못 입력됬습니다.");
					flag = true; 
				}
			} catch (NumberFormatException e) { // 정수형이 아니라면 예외처리하고 반복
				System.out.println("잘못 입력됬습니다. 숫자를 입력하세요.");
				flag = true;
			} // try catch 문자 정수 예외처리
		} while (flag);

		// 전화번호를 임시저장한 arrList에서 꺼낸다
		String selectkey = arrList.get(inputNum - 1);
	
		// 최종 선택한사람의 key=전화번호 리턴
		return selectkey;
	} // serchAddress

	
	// 수정 삭제 메시지 메소드
	
//	public String editAddress(String selectkey,HashMap<String, Address> myAddress,String input,Scanner scanner, AddressStream myStreamMethod, AddressMethodClub addressHelper) {
//		// 수정 삭제
//		if(input.equals("3")){
//			if (selectkey == null) { 
//				return;
//			} else { // 제대로 검색됬다면 회원 삭제
//				myAddress.remove(selectkey);
//				// 1에서 사용한 입력메소드 사용
//				addressHelper.inputAddress(scanner, myAddress, myStreamMethod);
//				System.out.println("수정이 완료되었습니다.");
//			}
//		} else if (input.equals("4")){
//			myAddress.remove(selectkey);
//			inputAddress(scanner, myAddress, myStreamMethod);
//			System.out.println("삭제가 완료되었습니다.");
//	  }
//
//	}
	

} //AddressMethodClub

