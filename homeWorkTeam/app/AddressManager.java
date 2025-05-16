package homeWorkTeam.app;

import java.util.HashMap;

import java.util.Scanner;

import homeWorkTeam.lib.Address;

import homeWorkTeam.lib.AddressMethodClub;

import homeWorkTeam.lib.AddressStream;

public class AddressManager {

	public static void main(String[] args) {

		//키보드 입력도 준비..

		Scanner scanner = new Scanner(System.in);

		//메뉴1~5입력할 input // 내클래스에 입력받을 변수 4개 :  이름, 전화, 주소, 그룹(구분)

		String input = null;

		//해시맵 준비 <전화번호, 내클래스>

		HashMap<String, Address> myAddress = new HashMap<>();

		// 입출력 클래스 불러오기 // 얘는 해시맵만 넣을 수 있다 // 로딩시 1회만 읽어서 해시맵에 대입하고 스트림도 닫는다.

		AddressStream myStreamMethod = new AddressStream();

		//빈 해시맵myAddress에 입력 메소드가 파일 읽어서 내용 채워줌. 파라미터에 넣는건 빈 해시맵, 받아올때는 내용 들어있는 것으로 받음

		myAddress = myStreamMethod.inputStream(myAddress);

		

		// 입력 삭제 등을 도와줄 외부 메소드

		AddressMethodClub addressHelper = new AddressMethodClub();

		

		while (true) { // 초기화면

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

			System.out.print("번호 선택: ");

			input = scanner.nextLine(); // 메뉴 입력 받음

			

			// 잘못입력시 재입력 요청

			if (!(input.equals("1") || input.equals("2") || input.equals("3") || input.equals("4")

					|| input.equals("5"))) {

				System.out.println("다시 입력해주세요.");

				// while : 스위치에 default 있을때는 continue; 사용. 그냥두면 switch-default는 항상 실행된다.

			} // 1~5 아닐경우 재입력 요청..

			if (input.equals("5")) {

				System.out.println("종료되었습니다.");

				break; // while 나가기 = App종료

			}

			

			switch (input) { // 메뉴선택

			case "1": // 입력

				// 입력 + 저장 + 파일 쓰기 메소드

				addressHelper.inputAddress(scanner, myAddress, myStreamMethod);

				break; // case-break : switch만 나간다 -> 처음으로 돌아간다.

				

			case "2": // 출력

				addressHelper.viewAllAddress(myAddress);

				break;

				

			case "3": // 검색 + 수정

				// 사람 찾기 메소드 (==> 검색한 사람의 key=전화번호를 리턴해줌)

				String selectkey = addressHelper.serchAddress(scanner, myAddress, input);

				//없는 회원이라면 null로 받아서 -> 메인화면으로 돌아간다. 이부분은 메소드에 안들어감..

				if (selectkey == null) {

					break;

				} else { // 제대로 검색됬다면 회원 삭제

					myAddress.remove(selectkey);

					// 1에서 사용한 입력메소드 사용

					addressHelper.inputAddress(scanner, myAddress, myStreamMethod);

					System.out.println("수정이 완료되었습니다.");

				}

				break;

				

			case "4": // 삭제

				// 사람 찾기 메소드

				selectkey = addressHelper.serchAddress(scanner, myAddress, input);

				//없는 회원이라면 null로 받아서 -> 메인화면으로 돌아간다. 이부분은 메소드에 안들어감..

				if (selectkey == null) {

					break;

				} else { // 제대로 검색됬다면 회원 삭제

					myAddress.remove(selectkey);

					// 해시맵을 파일에 쓰기

					myStreamMethod.makeAddressFile(myAddress);

					System.out.println("삭제가 완료되었습니다.");

				}

				break;

				

			} // switch

		} // while(true) 초기화면

		

		scanner.close();

	} // main end

	

} // class end