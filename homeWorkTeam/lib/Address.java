package homeWorkTeam.lib;
import java.io.Serializable;
public class Address implements Serializable {

	private static final long serialVersionUID = 1L;
	// 직렬화 버전관리 // 호환성 유지. 클래스 구조가 변경될때 이 값을 변경하지 않는다면 이전 버전의 객체를 새버전의 클래스로 역직렬화 시킨다.
	// 이값이 일치 하지 않으면 InvalidClassException 발생
	
	private String name;
	private String phone; //0으로 시작하고 싶기땜에 일단 문자로 저장
	private String address;
	private String group;
	
	public Address(String name, String phone, String address, String group) {
		this.name = name;
		this.phone = phone;
		this.address = address;
		this.group = group;
	}
	
	@Override
	public String toString() {
		return " 이름 : " + name + ", 전화번호 : " + phone + ", 주소 : "
				+ address + ", 분류 : " + group ;
	} // 내용 전체 출력해주는 메소드
	
	public String getName() { //이름 검색에 사용
		return name;
	}

}

