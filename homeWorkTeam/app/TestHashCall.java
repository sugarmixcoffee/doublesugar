package homeWorkTeam.app;

import java.util.HashMap;

import homeWorkTeam.lib.Address;
// 주소록 클래스가 들어간 해시맵이 작동하는지 테스트하는 app
public class TestHashCall {

	public static void main(String[] args) {
		HashMap<String, Address> myAddress = new HashMap<>();
		Address addr = new Address("고양이", "00099998888", "냥냥", "n333");
				
		System.out.println(myAddress);
		System.out.println(myAddress.containsValue("고양이"));
		System.out.println(addr.getName());
	}

}
