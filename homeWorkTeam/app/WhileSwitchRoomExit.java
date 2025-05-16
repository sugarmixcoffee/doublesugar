package homeWorkTeam.app;

import java.util.Scanner;

import homeWorkTeam.lib.ExitMethodTest;

public class WhileSwitchRoomExit {
// 메인 스위치문 탈출 테스트
	public static void main(String[] args) {
		ExitMethodTest goMain = new ExitMethodTest();
		
		while(true) {
			Scanner scanner = new Scanner(System.in);
			System.out.println("main ");
			String input =  scanner.nextLine();
			if (!(input.equals("1") || input.equals("2") || input.equals("3") )) {
				System.out.println("다시 입력해주세요.");
				continue; // 스위치에 디폴트가 있을 경우는 컨티뉴 해야 디폴트 안찍는다. 
				// break; // 여기서 브레이크는 전체 while 탈출
			}
			switch(input) {
			case "1":
				System.out.println(goMain);
				break;
			case "2":
				System.out.println("22222");
				break;
			default : // 1, 2, 조건 외에는 항상 실행된다. if에서 나와도 실행됨
				System.out.println("탈출");
				break;
			}// switch에서 break로는 스위치문만 탈출하게 된다. 
		}
		

	}

}
